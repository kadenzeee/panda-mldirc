#!/usr/bin/env python3

import sys, os, argparse

os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_ENABLE_XLA"] = "0"

import tensorflow as tf
import keras
import time
import datetime
import numpy as np
import subprocess

parser = argparse.ArgumentParser(
                    prog='prtai_film',
                    description='FiLM network for DIRC particle classification')

parser.add_argument('-s', '--s', help='Log compression fall-off scale factor')
parser.add_argument('-i', '--input', help='Data input file')
parser.add_argument('-o', '--output', help='Model output file', default='models')
args = parser.parse_args()

s = float(args.s)
infile = args.input
outfile = args.output

# Use float32 precision throughout to avoid quantization
tf.keras.mixed_precision.set_global_policy('float32')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.optimizer.set_jit(False)
        print("[INFO] Enabled GPU memory growth")
    except RuntimeError as e:
        print(e)

os.environ["ROBCLAS_VERBOSE"] = "0"
os.environ["ROCM_INFO_LEVEL"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print(f"[INFO] Environment set")

program_start = time.time()



print("[INFO] Loading .dat from ", infile)

# ------------------------------------------------
# SET THESE PARAMS
# ------------------------------------------------

nevents = 200000
time_dim = 8*64
hist_dim = 10*8
angle_dim = 7

# ------------------------------------------------

TIMES = np.memmap(f"{infile}/TIMES_full.dat",  dtype=np.float16, mode='r', shape=(nevents, time_dim))
HISTS = np.memmap(f"{infile}/HISTS_full.dat",  dtype=np.int8,    mode='r', shape=(nevents, hist_dim))
ANGLES = np.memmap(f"{infile}/ANGLES_full.dat", dtype=np.float16, mode='r', shape=(nevents, angle_dim))
LABELS = np.memmap(f"{infile}/LABELS_full.dat", dtype=np.int8,    mode='r', shape=(nevents,))

print(f"[INFO] Data size: {sys.getsizeof(TIMES)//10**6} MB")

print(LABELS[0:100])

# ---------------------------------------------------------------
#
#                       PARAMETERS
#
# ---------------------------------------------------------------
class_names = ['Pi+', 'Kaon+']
num_classes = len(class_names) # Pions or kaons?


batch_size  = 1024 # How many events to feed to NN at a time?
nepochs     = 20 # How many epochs?

trainfrac   = 0.70
valfrac     = 0.15
testfrac    = 0.15

datafrac    = 1  # What fraction of data to use?
# ---------------------------------------------------------------

trainfrac, valfrac, testfrac = (datafrac*trainfrac, datafrac*valfrac, datafrac*testfrac)

trainend    = int(np.floor(nevents * trainfrac))
valend      = int(trainend + np.floor(nevents * valfrac))
testend     = int(valend + np.floor(nevents * testfrac))

print(f"[INFO] Using {testend} out of {nevents} available events")

traintimes  = TIMES[:trainend]
trainhists  = HISTS[:trainend]
trainangles = ANGLES[:trainend]
trainlabels = LABELS[:trainend]
valtimes    = TIMES[trainend:valend]
valhists    = HISTS[trainend:valend]
valangles   = ANGLES[trainend:valend]
vallabels   = LABELS[trainend:valend]
testtimes   = TIMES[valend:testend]
testhists   = HISTS[valend:testend]
testangles  = ANGLES[valend:testend]
testlabels  = LABELS[valend:testend]


class ScheduledFiLM(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ScheduledFiLM, self).__init__(**kwargs)
        self.lambda_var = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.ranp_rate = 0.01
        

    def call(self, inputs):
        x, gamma, beta = inputs
        lam = tf.clip_by_value(self.lambda_var, 0.0, 1.0)
        return (1.0 + lam * gamma) * x + lam * beta

class ScheduledFiLMCallback(keras.callbacks.Callback):
    def __init__(self, film_layer, nepochs):
        super(ScheduledFiLMCallback, self).__init__()
        self.film_layer = film_layer
        self.nepochs = nepochs

    def on_epoch_end(self, epoch, logs=None):
        new_lambda = (2*epoch + self.nepochs) / (epoch + self.nepochs) - 0.5
        self.film_layer.lambda_var.assign(new_lambda)
        print(f'\nUpdated FiLM lambda to {new_lambda:.4f}')

class BatchGenerator(keras.utils.Sequence):
    """
    Converts to dense batches for training during runtime.
    Class keeps ordering and supports shuffling each epoch.

    
    Parameters
    ----------
    *args : ndarray
        Sparse or dense matrices to be placed into batches. The arguments should include label data as the last argument.
    >>> train_gen = BatchGenerator(times, angles, labels, *kwargs)
    batch_size : int
        Number of matrices to batch.
    shuffle : bool
        Whether or not to shuffle data on epoch end.
    """

    def __init__(self, *args, batch_size=256, shuffle=True):
        self.args = args
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = self.args[0].shape[0]
        self.indices = np.arange(self.n)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.n / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_list = [np.asarray(data[batch_idx]) for data in self.args]
        
        inputs = batch_list[:-1]
        labels = batch_list[-1]
        
        return tuple(inputs), labels
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

AUTOTUNE = tf.data.AUTOTUNE

def make_dataset(times, hists, angles, labels, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(
        ((times, hists, angles), labels)
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=100_000)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)

    return ds


def width_function(x, N_max, x_min):
    return N_max * np.log(1.0 / x) / np.log(1.0 / x_min)

def layer_dims(s, L, N_max, N_min=8):
    x_min = 1.0 / (L + 1)
    xs = ((np.arange(L) + 1) / (L + 1)) ** s

    widths = width_function(xs, N_max, x_min)
    widths = np.maximum(widths.astype(int), N_min)

    return widths

L = 5
N_max = 256

widths = layer_dims(s, L, N_max)
print(widths)

dropout = 0.1

# Time Data Branch
time_input = keras.Input(shape=(time_dim,))

t = keras.layers.Dense(widths[0], activation='gelu')(time_input)
t = keras.layers.Dense(widths[1], activation='gelu')(t)
t = keras.layers.Dense(widths[2], activation='gelu')(t)

# Histogram Data Branch
@keras.utils.register_keras_serializable(package='Custom', name='ScaleHistograms')
class ScaleHistograms(keras.layers.Layer):
    def __init__(self, initial_scale=1.0, dtype='bfloat16', **kwargs):
        super().__init__(**kwargs)
        self.initial_scale = initial_scale
        self._dtype = dtype

    def build(self, input_shape):
        # trainable scalar weight tracked by the layer
        self.scale = self.add_weight(
            name='scale',
            shape=(),
            initializer=tf.keras.initializers.Constant(self.initial_scale),
            trainable=True,
            dtype=self._dtype
        )

    def call(self, x):
        # ensure same dtype for multiplication
        s = tf.cast(self.scale, x.dtype)
        return x * s

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'initial_scale': self.initial_scale, 'dtype': self._dtype})
        return cfg

hist_input = keras.Input(shape=(hist_dim,))
h = keras.layers.LayerNormalization()(hist_input)
h = keras.layers.Dense(widths[2]//2, activation='gelu')(h)

# Angle Data Branch
@keras.utils.register_keras_serializable(package='Custom', name='ScaleAngles')
class ScaleAngles(keras.layers.Layer):
    def __init__(self, initial_scale=1.0, dtype='bfloat16', **kwargs):
        super().__init__(**kwargs)
        self.initial_scale = initial_scale
        self._dtype = dtype

    def build(self, input_shape):
        # trainable scalar weight tracked by the layer
        self.scale = self.add_weight(
            name='scale',
            shape=(),
            initializer=tf.keras.initializers.Constant(self.initial_scale),
            trainable=True,
            dtype=self._dtype
        )

    def call(self, x):
        # ensure same dtype for multiplication
        s = tf.cast(self.scale, x.dtype)
        angle = x[:, 4:] * s
        rest = x[:, :4]
        return tf.concat([angle, rest], axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'initial_scale': self.initial_scale, 'dtype': self._dtype})
        return cfg

angle_input = keras.Input(shape=(angle_dim,))
scaled_input = ScaleAngles(initial_scale=1.0, dtype='bfloat16')(angle_input)

a = keras.layers.Dense(widths[3], activation='gelu')(scaled_input)
a = keras.layers.Dense(widths[2], activation='gelu')(a)

# Produce FiLM parameters for FiLM layer 1
gamma = keras.layers.Dense(widths[2], kernel_regularizer=keras.regularizers.L2(1e-04), name='gamma1')(a)
beta  = keras.layers.Dense(widths[2], activation='linear', name='beta1')(a)

# FiLM layer 1
t_mod = t * gamma + beta

# Combined timing and angle
drop = keras.layers.Dropout(dropout, name='dropout')(t_mod)
x = keras.layers.Dense(widths[2], activation='gelu')(drop)

# Produce FiLM parameters for FiLM layer 2
gamma = keras.layers.Dense(widths[2], kernel_regularizer=keras.regularizers.L2(1e-04), name='gamma2')(h)
beta  = keras.layers.Dense(widths[2], activation='linear', name='beta2')(h)

# FiLM layer 2
x_mod = x * gamma + beta


# Output layers
drop = keras.layers.Dropout(dropout, name='dropout2')(x_mod)
x = keras.layers.Dense(widths[3], activation='gelu')(drop)
x = keras.layers.Dense(widths[4], activation='gelu')(x)
out = keras.layers.Dense(num_classes, activation='softmax')(x)


model = keras.Model(inputs=[time_input, hist_input, angle_input], outputs=out)
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
#model.summary()


train_ds = make_dataset(traintimes, trainhists, trainangles, trainlabels, batch_size)
val_ds   = make_dataset(valtimes, valhists, valangles, vallabels, batch_size, shuffle=False)
test_ds  = make_dataset(testtimes, testhists, testangles, testlabels, batch_size, shuffle=False)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=nepochs, 
    validation_freq=4
    #callbacks=[ScheduledFiLMCallback(film, nepochs)],
)

test_loss, test_acc = model.evaluate(
    test_ds, verbose=2
)

print('\nTest accuracy:', test_acc)
print('Test loss:', test_loss)

subprocess.run(f'mkdir -p {outfile}', shell=True)
date_str = datetime.date.today().isoformat()
i = 1
while True:
    out_name = f"{outfile}/{date_str}_model{i}_s{s}.keras"
    if not os.path.exists(out_name):
        break
    i += 1

# Convert model to float32 before saving to avoid quantization
model = model.astype('float32') if hasattr(model, 'astype') else model
model.save(out_name)
print(f"Saved model to {out_name}")

program_end = time.time()
print(f"Done in {program_end - program_start}s")
