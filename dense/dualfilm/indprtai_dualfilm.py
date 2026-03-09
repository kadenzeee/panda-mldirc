#!/usr/bin/env python3

import sys, os, argparse

os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_ENABLE_XLA"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ROCM_FUSION_ENABLE"] = "0"
os.environ["TF_MIOPEN_USE_TENSOR_OPS"] = "0"

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

print(f"[INFO] Environment set")

program_start = time.time()



print("[INFO] Loading .dat from ", infile)

# ------------------------------------------------
# SET THESE PARAMS
# ------------------------------------------------

nevents = 19940000
max_photons = 128
hist_dim = 10*8
angle_dim = 7

# ------------------------------------------------

TIMES = np.memmap(f"{infile}/TIMES_full.dat",  dtype=np.float16, mode='r', shape=(nevents, max_photons, 2))
HISTS = np.memmap(f"{infile}/HISTS_full.dat",  dtype=np.int8,    mode='r', shape=(nevents, hist_dim))
ANGLES = np.memmap(f"{infile}/ANGLES_full.dat", dtype=np.float16, mode='r', shape=(nevents, angle_dim))
LABELS = np.memmap(f"{infile}/LABELS_full.dat", dtype=np.int8,    mode='r', shape=(nevents,))

print(f"[INFO] Data size: {sys.getsizeof(TIMES)//10**6} MB")

print(TIMES[0,:10])
print(np.min(TIMES), np.max(TIMES))
empty_events = np.where(np.sum(HISTS, axis=1) == 0)[0]
print(empty_events)

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

datafrac    = 0.3  # What fraction of data to use?
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


AUTOTUNE = tf.data.AUTOTUNE

def make_dataset(times, hists, angles, labels, batch_size, shuffle=True):
    
    def cast_fn(x, y):
        t, h, a = x
        return(tf.cast(t, tf.float32),
               tf.cast(h, tf.float32),
               tf.cast(a, tf.float32)), y
    
    
    ds = tf.data.Dataset.from_tensor_slices(
        ((times, hists, angles), labels)
    )

    ds = ds.map(cast_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=20_000)

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
time_input = keras.Input(shape=(max_photons, 2))
time_input = keras.layers.LayerNormalization()(time_input)

mask = keras.layers.Lambda(
    lambda x: tf.reduce_sum(tf.abs(x), axis=-1) > 0,
    output_shape=(max_photons,)
)(time_input)

t = keras.layers.Dense(widths[1], activation='gelu')(time_input)
t = keras.layers.Dense(widths[2], activation='gelu')(t)

t = keras.layers.GlobalAveragePooling1D()(t, mask=mask)

# Histogram Data Branch
@keras.utils.register_keras_serializable(package='Custom', name='ScaleHistograms')
class ScaleHistograms(keras.layers.Layer):
    def __init__(self, initial_scale=1.0, dtype='float32', **kwargs):
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
    def __init__(self, initial_scale=1.0, dtype='float32', **kwargs):
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
scaled_input = ScaleAngles(initial_scale=1.0, dtype='float32')(angle_input)

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
model.summary()


train_ds = make_dataset(traintimes, trainhists, trainangles, trainlabels, batch_size, shuffle=False)
val_ds   = make_dataset(valtimes, valhists, valangles, vallabels, batch_size, shuffle=False)
test_ds  = make_dataset(testtimes, testhists, testangles, testlabels, batch_size, shuffle=False)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=nepochs, 
    validation_freq=1
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


model.save(out_name)
print(f"Saved model to {out_name}")

program_end = time.time()
print(f"Done in {program_end - program_start}s")
