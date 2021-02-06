import tensorflow as tf
import tensorflow_datasets as tfds
import math

#Downloading dataset from tensorflow_datasets
camelyon = tfds.load('patch_camelyon', as_supervised=True, shuffle_files=True)

# Image tensor values will be in range 0-1
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train = camelyon['train'].shuffle(1000).batch(32).map(preprocess).prefetch(tf.data.experimental.AUTOTUNE)
val = camelyon['validation'].batch(32).map(preprocess).prefetch(tf.data.experimental.AUTOTUNE)
test = camelyon['test'].batch(32).map(preprocess).prefetch(tf.data.experimental.AUTOTUNE)

#Let's build the model
from tensorflow.keras import layers


# Defining the first and second stage layers.
def stage_1_and_2(x):
    x = layers.Conv2D(64, (7, 7), strides=2, padding='SAME', activation='relu',
                      kernel_initializer=tf.keras.initializers.glorot_normal,
                      bias_initializer=tf.keras.initializers.Constant(value=0.2))(x)

    x = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="SAME")(x)

    x = layers.Conv2D(64, (1, 1), strides=1, padding='SAME', activation='relu',
                      kernel_initializer=tf.keras.initializers.glorot_normal,
                      bias_initializer=tf.keras.initializers.Constant(value=0.2))(x)

    x = layers.Conv2D(192, (3, 3), strides=1, padding='SAME', activation='relu',
                      kernel_initializer=tf.keras.initializers.glorot_normal,
                      bias_initializer=tf.keras.initializers.Constant(value=0.2))(x)

    x = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="SAME")(x)

    return x


# Defining the inception layers to be put in stages 3, 4 and 5.

def inception_st3(x, filter_sizes, reduction_sizes):
    first_branch = layers.Conv2D(filter_sizes[0], (1, 1), padding='SAME', activation='relu',
                                 kernel_initializer=tf.keras.initializers.glorot_normal,
                                 bias_initializer=tf.keras.initializers.Constant(value=0.2))(x)

    second_branch_red = layers.Conv2D(reduction_sizes[0], (1, 1), padding='SAME', activation='relu',
                                      kernel_initializer=tf.keras.initializers.glorot_normal,
                                      bias_initializer=tf.keras.initializers.Constant(value=0.2))(x)

    second_branch = layers.Conv2D(filter_sizes[1], (3, 3), padding='SAME', activation='relu',
                                  kernel_initializer=tf.keras.initializers.glorot_normal,
                                  bias_initializer=tf.keras.initializers.Constant(value=0.2))(second_branch_red)

    third_branch_red = layers.Conv2D(reduction_sizes[1], (1, 1), padding='SAME', activation='relu',
                                     kernel_initializer=tf.keras.initializers.glorot_normal,
                                     bias_initializer=tf.keras.initializers.Constant(value=0.2))(x)

    third_branch = layers.Conv2D(filter_sizes[2], (3, 3), padding='SAME', activation='relu',
                                 kernel_initializer=tf.keras.initializers.glorot_normal,
                                 bias_initializer=tf.keras.initializers.Constant(value=0.2))(third_branch_red)

    third_branch = layers.Conv2D(filter_sizes[2], (3, 3), padding='SAME', activation='relu',
                                 kernel_initializer=tf.keras.initializers.glorot_normal,
                                 bias_initializer=tf.keras.initializers.Constant(value=0.2))(third_branch)

    fourth_branch = layers.MaxPool2D(pool_size=(3, 3), strides=1, padding="SAME")(x)

    fourth_branch = layers.Conv2D(filter_sizes[3], (1, 1), padding='SAME', activation='relu',
                                  kernel_initializer=tf.keras.initializers.glorot_normal,
                                  bias_initializer=tf.keras.initializers.Constant(value=0.2))(fourth_branch)

    return layers.concatenate([first_branch, second_branch, third_branch, fourth_branch], axis=3)


def inception_st4(x, filter_sizes, reduction_sizes):
    first_branch = layers.Conv2D(filter_sizes[0], (1, 1), padding='SAME', activation='relu',
                                 kernel_initializer=tf.keras.initializers.glorot_normal,
                                 bias_initializer=tf.keras.initializers.Constant(value=0.2))(x)

    second_branch_red = layers.Conv2D(reduction_sizes[0], (1, 1), padding='SAME', activation='relu',
                                      kernel_initializer=tf.keras.initializers.glorot_normal,
                                      bias_initializer=tf.keras.initializers.Constant(value=0.2))(x)

    second_branch = layers.Conv2D(filter_sizes[1], (1, 7), padding='SAME', activation='relu',
                                  kernel_initializer=tf.keras.initializers.glorot_normal,
                                  bias_initializer=tf.keras.initializers.Constant(value=0.2))(second_branch_red)

    second_branch = layers.Conv2D(filter_sizes[1], (7, 1), padding='SAME', activation='relu',
                                  kernel_initializer=tf.keras.initializers.glorot_normal,
                                  bias_initializer=tf.keras.initializers.Constant(value=0.2))(second_branch)

    third_branch_red = layers.Conv2D(reduction_sizes[1], (1, 1), padding='SAME', activation='relu',
                                     kernel_initializer=tf.keras.initializers.glorot_normal,
                                     bias_initializer=tf.keras.initializers.Constant(value=0.2))(x)

    third_branch = layers.Conv2D(filter_sizes[2], (1, 7), padding='SAME', activation='relu',
                                 kernel_initializer=tf.keras.initializers.glorot_normal,
                                 bias_initializer=tf.keras.initializers.Constant(value=0.2))(third_branch_red)

    third_branch = layers.Conv2D(filter_sizes[2], (7, 1), padding='SAME', activation='relu',
                                 kernel_initializer=tf.keras.initializers.glorot_normal,
                                 bias_initializer=tf.keras.initializers.Constant(value=0.2))(third_branch)

    third_branch = layers.Conv2D(filter_sizes[2], (1, 7), padding='SAME', activation='relu',
                                 kernel_initializer=tf.keras.initializers.glorot_normal,
                                 bias_initializer=tf.keras.initializers.Constant(value=0.2))(third_branch)

    third_branch = layers.Conv2D(filter_sizes[2], (7, 1), padding='SAME', activation='relu',
                                 kernel_initializer=tf.keras.initializers.glorot_normal,
                                 bias_initializer=tf.keras.initializers.Constant(value=0.2))(third_branch)

    fourth_branch = layers.MaxPool2D(pool_size=(3, 3), strides=1, padding="SAME")(x)

    fourth_branch = layers.Conv2D(filter_sizes[3], (1, 1), padding='SAME', activation='relu',
                                  kernel_initializer=tf.keras.initializers.glorot_normal,
                                  bias_initializer=tf.keras.initializers.Constant(value=0.2))(fourth_branch)

    return layers.concatenate([first_branch, second_branch, third_branch, fourth_branch], axis=3)


def inception_st5(x, filter_sizes, reduction_sizes):
    first_branch = layers.Conv2D(filter_sizes[0], (1, 1), padding='SAME', activation='relu',
                                 kernel_initializer=tf.keras.initializers.glorot_normal,
                                 bias_initializer=tf.keras.initializers.Constant(value=0.2))(x)

    second_branch_red = layers.Conv2D(reduction_sizes[0], (1, 1), padding='SAME', activation='relu',
                                      kernel_initializer=tf.keras.initializers.glorot_normal,
                                      bias_initializer=tf.keras.initializers.Constant(value=0.2))(x)

    second_branch_1 = layers.Conv2D(int(filter_sizes[1] / 2), (3, 1), padding='SAME', activation='relu',
                                    kernel_initializer=tf.keras.initializers.glorot_normal,
                                    bias_initializer=tf.keras.initializers.Constant(value=0.2))(second_branch_red)

    second_branch_2 = layers.Conv2D(int(filter_sizes[1] / 2), (1, 3), padding='SAME', activation='relu',
                                    kernel_initializer=tf.keras.initializers.glorot_normal,
                                    bias_initializer=tf.keras.initializers.Constant(value=0.2))(second_branch_red)

    third_branch_red = layers.Conv2D(reduction_sizes[1], (1, 1), padding='SAME', activation='relu',
                                     kernel_initializer=tf.keras.initializers.glorot_normal,
                                     bias_initializer=tf.keras.initializers.Constant(value=0.2))(x)

    third_branch = layers.Conv2D(filter_sizes[2], (3, 3), padding='SAME', activation='relu',
                                 kernel_initializer=tf.keras.initializers.glorot_normal,
                                 bias_initializer=tf.keras.initializers.Constant(value=0.2))(third_branch_red)

    third_branch_1 = layers.Conv2D(int(filter_sizes[2] / 2), (3, 1), padding='SAME', activation='relu',
                                   kernel_initializer=tf.keras.initializers.glorot_normal,
                                   bias_initializer=tf.keras.initializers.Constant(value=0.2))(third_branch)

    third_branch_2 = layers.Conv2D(int(filter_sizes[2] / 2), (1, 3), padding='SAME', activation='relu',
                                   kernel_initializer=tf.keras.initializers.glorot_normal,
                                   bias_initializer=tf.keras.initializers.Constant(value=0.2))(third_branch)

    fourth_branch = layers.MaxPool2D(pool_size=(3, 3), strides=1, padding="SAME")(x)

    fourth_branch = layers.Conv2D(filter_sizes[3], (1, 1), padding='SAME', activation='relu',
                                  kernel_initializer=tf.keras.initializers.glorot_normal,
                                  bias_initializer=tf.keras.initializers.Constant(value=0.2))(fourth_branch)

    return layers.concatenate(
        [first_branch, second_branch_1, second_branch_2, third_branch_1, third_branch_2, fourth_branch], axis=3)

# Defining the rest of stages

def stage_3(x):
  filter_sizes = [[64, 128, 32, 32],[128, 192, 96, 64]]
  reduction_sizes = [[96, 16],[128, 32]]
  for i in range(2):
    x = inception_st3(x, filter_sizes[i], reduction_sizes[i])
  return x

def stage_4(x):
  filter_sizes = [[192, 208, 48, 64], [160, 224, 64, 64],[128, 256, 64, 64],
                  [112, 288, 64, 64], [256, 320, 128, 128]]
  reduction_sizes = [[96, 16],[112, 24],[128, 24],[144, 32],[160, 32]]
  for i in range(5):
    x = inception_st4(x, filter_sizes[i], reduction_sizes[i])
    if i == 0:
      aux_1 = layers.AveragePooling2D(5, strides=3)(x)
      aux_1 = layers.Conv2D(128, (1, 1), padding='SAME', activation='relu')(aux_1)
      aux_1 = layers.Flatten()(aux_1)
      aux_1 = layers.Dense(1024, activation='relu')(aux_1)
      aux_1 = layers.Dropout(0.7)(aux_1)
      aux_1 = layers.Dense(1, activation='sigmoid', name='aux_output_1')(aux_1)
    elif i == 3:
      aux_2 = layers.AveragePooling2D(5, strides=3)(x)
      aux_2 = layers.Conv2D(128, (1, 1), padding='SAME', activation='relu')(aux_2)
      aux_2 = layers.Flatten()(aux_2)
      aux_2 = layers.Dense(1024, activation='relu')(aux_2)
      aux_2 = layers.Dropout(0.7)(aux_2)
      aux_2 = layers.Dense(1, activation='sigmoid', name='aux_output_2')(aux_2)

  return x, aux_1, aux_2

def stage_5(x):
  filter_sizes = [[256, 320, 128, 128],[384, 384, 128, 128]]
  reduction_sizes = [[160, 32],[192,48]]
  for i in range(2):
    x = inception_st5(x, filter_sizes[i], reduction_sizes[i])
  return x

# Define a dummy input, with img dimensions, in order to build the model
inp = layers.Input(shape=(96, 96, 3))

def build_model():
  x = stage_1_and_2(inp)
  x = stage_3(x)
  x = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="SAME")(x)
  x, aux_1, aux_2 = stage_4(x)
  x = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="SAME")(x)
  x = stage_5(x)
  x = layers.GlobalAveragePooling2D()(x)
  x = layers.Dropout(0.4)(x)
  x = layers.Dense(1, activation='sigmoid', name='main_output')(x)
  return tf.keras.Model(inp, [x, aux_1, aux_2])

model = build_model()

from tensorflow.keras import losses, optimizers, metrics, callbacks

# Defining the learning rate drop

epochs = 100
initial_lrate = 0.01

def decay(epoch, steps=100):
    initial_lrate = 0.01
    drop = 0.96
    epochs_drop = 8
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

sgd = optimizers.SGD(lr=initial_lrate, momentum=0.9, nesterov=False)


lr_sc = callbacks.LearningRateScheduler(decay, verbose=1)

call_backs = [
             callbacks.EarlyStopping(monitor='val_main_output_binary_accuracy', patience=3, restore_best_weights=True),
             callbacks.TerminateOnNaN(),
             lr_sc
            ]

model.compile(loss=['binary_crossentropy', 'binary_crossentropy',
                    'binary_crossentropy'], loss_weights=[1, 0.3, 0.3],
                    optimizer=sgd, metrics=['binary_accuracy'])

model.fit(train, epochs=epochs, validation_data=val, callbacks=call_backs)
model.evaluate(test)