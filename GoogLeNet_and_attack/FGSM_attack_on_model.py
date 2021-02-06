import tensorflow as tf
import tensorflow_datasets as tfds
import math

camelyon = tfds.load('patch_camelyon', as_supervised=True, shuffle_files=True)

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train = camelyon['train'].shuffle(1000).batch(32).map(preprocess).prefetch(tf.data.experimental.AUTOTUNE)
val = camelyon['validation'].batch(32).map(preprocess).prefetch(tf.data.experimental.AUTOTUNE)
test = camelyon['test'].batch(32).map(preprocess).prefetch(tf.data.experimental.AUTOTUNE)



# Implementing inception layers
from tensorflow.keras import layers


# Implementing inception layers

class Inception_st3(tf.keras.layers.Layer):
    def __init__(self, filters_version=0):
        super(Inception_st3, self).__init__()
        self.filter_sizes = [[64, 128, 32, 32], [128, 192, 96, 64]][filters_version]
        self.reduction_sizes = [[96, 16], [128, 32]][filters_version]
        self.first_fil = layers.Conv2D(self.filter_sizes[0], (1, 1), padding='SAME', activation='relu',
                                       kernel_initializer=tf.keras.initializers.glorot_normal,
                                       bias_initializer=tf.keras.initializers.Constant(value=0.2))
        self.second_red = layers.Conv2D(self.reduction_sizes[0], (1, 1), padding='SAME', activation='relu',
                                        kernel_initializer=tf.keras.initializers.glorot_normal,
                                        bias_initializer=tf.keras.initializers.Constant(value=0.2))
        self.second_fil = layers.Conv2D(self.filter_sizes[1], (3, 3), padding='SAME', activation='relu',
                                        kernel_initializer=tf.keras.initializers.glorot_normal,
                                        bias_initializer=tf.keras.initializers.Constant(value=0.2))
        self.third_red = layers.Conv2D(self.reduction_sizes[1], (1, 1), padding='SAME', activation='relu',
                                       kernel_initializer=tf.keras.initializers.glorot_normal,
                                       bias_initializer=tf.keras.initializers.Constant(value=0.2))
        self.third_fil = layers.Conv2D(self.filter_sizes[2], (3, 3), padding='SAME', activation='relu',
                                       kernel_initializer=tf.keras.initializers.glorot_normal,
                                       bias_initializer=tf.keras.initializers.Constant(value=0.2))
        self.third_fil_2 = layers.Conv2D(self.filter_sizes[2], (3, 3), padding='SAME', activation='relu',
                                         kernel_initializer=tf.keras.initializers.glorot_normal,
                                         bias_initializer=tf.keras.initializers.Constant(value=0.2))
        self.MAX = layers.MaxPool2D(pool_size=(3, 3), strides=1, padding="SAME")
        self.fourth_fil = layers.Conv2D(self.filter_sizes[3], (1, 1), padding='SAME', activation='relu',
                                        kernel_initializer=tf.keras.initializers.glorot_normal,
                                        bias_initializer=tf.keras.initializers.Constant(value=0.2))

    def call(self, inputs):
        first_branch = self.first_fil(inputs)

        second_branch_red = self.second_red(inputs)

        second_branch = self.second_fil(second_branch_red)

        third_branch_red = self.third_red(inputs)

        third_branch = self.third_fil(third_branch_red)

        third_branch = self.third_fil_2(third_branch)

        fourth_branch = self.MAX(inputs)

        fourth_branch = self.fourth_fil(fourth_branch)

        return layers.concatenate([first_branch, second_branch, third_branch, fourth_branch], axis=3)

class Inception_st4(tf.keras.layers.Layer):
    def __init__(self, filters_version=0):
        super(Inception_st4, self).__init__()
        self.filter_sizes = [[192, 208, 48, 64], [160, 224, 64, 64], [128, 256, 64, 64],
                             [112, 288, 64, 64], [256, 320, 128, 128]][filters_version]
        self.reduction_sizes = [[96, 16], [112, 24], [128, 24], [144, 32], [160, 32]][filters_version]
        self.first_fil = layers.Conv2D(self.filter_sizes[0], (1, 1), padding='SAME', activation='relu',
                                       kernel_initializer=tf.keras.initializers.glorot_normal,
                                       bias_initializer=tf.keras.initializers.Constant(value=0.2))
        self.second_red = layers.Conv2D(self.reduction_sizes[0], (1, 1), padding='SAME', activation='relu',
                                        kernel_initializer=tf.keras.initializers.glorot_normal,
                                        bias_initializer=tf.keras.initializers.Constant(value=0.2))
        self.second_fil = layers.Conv2D(self.filter_sizes[1], (1, 7), padding='SAME', activation='relu',
                                        kernel_initializer=tf.keras.initializers.glorot_normal,
                                        bias_initializer=tf.keras.initializers.Constant(value=0.2))
        self.second_fil_1 = layers.Conv2D(self.filter_sizes[1], (7, 1), padding='SAME', activation='relu',
                                          kernel_initializer=tf.keras.initializers.glorot_normal,
                                          bias_initializer=tf.keras.initializers.Constant(value=0.2))
        self.third_red = layers.Conv2D(self.reduction_sizes[1], (1, 1), padding='SAME', activation='relu',
                                       kernel_initializer=tf.keras.initializers.glorot_normal,
                                       bias_initializer=tf.keras.initializers.Constant(value=0.2))
        self.third_fil = layers.Conv2D(self.filter_sizes[2], (1, 7), padding='SAME', activation='relu',
                                       kernel_initializer=tf.keras.initializers.glorot_normal,
                                       bias_initializer=tf.keras.initializers.Constant(value=0.2))
        self.third_fil_1 = layers.Conv2D(self.filter_sizes[2], (7, 1), padding='SAME', activation='relu',
                                         kernel_initializer=tf.keras.initializers.glorot_normal,
                                         bias_initializer=tf.keras.initializers.Constant(value=0.2))
        self.third_fil_2 = layers.Conv2D(self.filter_sizes[2], (1, 7), padding='SAME', activation='relu',
                                         kernel_initializer=tf.keras.initializers.glorot_normal,
                                         bias_initializer=tf.keras.initializers.Constant(value=0.2))
        self.third_fil_3 = layers.Conv2D(self.filter_sizes[2], (7, 1), padding='SAME', activation='relu',
                                         kernel_initializer=tf.keras.initializers.glorot_normal,
                                         bias_initializer=tf.keras.initializers.Constant(value=0.2))
        self.MAX = layers.MaxPool2D(pool_size=(3, 3), strides=1, padding="SAME")
        self.fourth_fil = layers.Conv2D(self.filter_sizes[3], (1, 1), padding='SAME', activation='relu',
                                        kernel_initializer=tf.keras.initializers.glorot_normal,
                                        bias_initializer=tf.keras.initializers.Constant(value=0.2))

    def call(self, inputs):
        first_branch = self.first_fil(inputs)

        second_branch_red = self.second_red(inputs)

        second_branch = self.second_fil(second_branch_red)

        second_branch = self.second_fil_1(second_branch)

        third_branch_red = self.third_red(inputs)

        third_branch = self.third_fil(third_branch_red)

        third_branch = self.third_fil_1(third_branch)

        third_branch = self.third_fil_2(third_branch)

        third_branch = self.third_fil_3(third_branch)

        fourth_branch = self.MAX(inputs)

        fourth_branch = self.fourth_fil(fourth_branch)

        return layers.concatenate([first_branch, second_branch, third_branch, fourth_branch], axis=3)


class Inception_st5(tf.keras.layers.Layer):
    def __init__(self, filters_version=0):
        super(Inception_st5, self).__init__()
        self.filter_sizes = [[256, 320, 128, 128], [384, 384, 128, 128]][filters_version]
        self.reduction_sizes = [[160, 32], [192, 48]][filters_version]
        self.first_fil = layers.Conv2D(self.filter_sizes[0], (1, 1), padding='SAME', activation='relu',
                                       kernel_initializer=tf.keras.initializers.glorot_normal,
                                       bias_initializer=tf.keras.initializers.Constant(value=0.2))
        self.second_red = layers.Conv2D(self.reduction_sizes[0], (1, 1), padding='SAME', activation='relu',
                                        kernel_initializer=tf.keras.initializers.glorot_normal,
                                        bias_initializer=tf.keras.initializers.Constant(value=0.2))
        self.second_fil = layers.Conv2D(int(self.filter_sizes[1] / 2), (3, 1), padding='SAME', activation='relu',
                                        kernel_initializer=tf.keras.initializers.glorot_normal,
                                        bias_initializer=tf.keras.initializers.Constant(value=0.2))
        self.second_fil_1 = layers.Conv2D(int(self.filter_sizes[1] / 2), (1, 3), padding='SAME', activation='relu',
                                          kernel_initializer=tf.keras.initializers.glorot_normal,
                                          bias_initializer=tf.keras.initializers.Constant(value=0.2))
        self.third_red = layers.Conv2D(self.reduction_sizes[1], (1, 1), padding='SAME', activation='relu',
                                       kernel_initializer=tf.keras.initializers.glorot_normal,
                                       bias_initializer=tf.keras.initializers.Constant(value=0.2))
        self.third_fil = layers.Conv2D(self.filter_sizes[2], (3, 3), padding='SAME', activation='relu',
                                       kernel_initializer=tf.keras.initializers.glorot_normal,
                                       bias_initializer=tf.keras.initializers.Constant(value=0.2))
        self.third_fil_1 = layers.Conv2D(int(self.filter_sizes[2] / 2), (3, 1), padding='SAME', activation='relu',
                                         kernel_initializer=tf.keras.initializers.glorot_normal,
                                         bias_initializer=tf.keras.initializers.Constant(value=0.2))
        self.third_fil_2 = layers.Conv2D(int(self.filter_sizes[2] / 2), (1, 3), padding='SAME', activation='relu',
                                         kernel_initializer=tf.keras.initializers.glorot_normal,
                                         bias_initializer=tf.keras.initializers.Constant(value=0.2))
        self.MAX = layers.MaxPool2D(pool_size=(3, 3), strides=1, padding="SAME")
        self.fourth_fil = layers.Conv2D(self.filter_sizes[3], (1, 1), padding='SAME', activation='relu',
                                        kernel_initializer=tf.keras.initializers.glorot_normal,
                                        bias_initializer=tf.keras.initializers.Constant(value=0.2))

    def call(self, inputs):
        first_branch = self.first_fil(inputs)

        second_branch_red = self.second_red(inputs)

        second_branch_1 = self.second_fil(second_branch_red)

        second_branch_2 = self.second_fil_1(second_branch_red)

        third_branch_red = self.third_red(inputs)

        third_branch = self.third_fil(third_branch_red)

        third_branch_1 = self.third_fil_1(third_branch)

        third_branch_2 = self.third_fil_2(third_branch)

        fourth_branch = self.MAX(inputs)

        fourth_branch = self.fourth_fil(fourth_branch)

        return layers.concatenate(
            [first_branch, second_branch_1, second_branch_2, third_branch_1, third_branch_2, fourth_branch], axis=3)

# These components are simply to be put inside a sequential model. Since in GoogLeNet
# inception layers of the same stages have different filters sizes for each inner layer,
# you can select the appropriate set of sizes by passing an integer from 0 as only
# user-dependant argument.

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import TensorFlowV2Classifier
import numpy as np

def separate(x):
  labs = []
  imgs = []
  for images, labels in x:
    imgs.append(images)
    labs.append(labels)
  return tf.convert_to_tensor(np.stack(imgs)), tf.convert_to_tensor(np.array(labs))

# Due to memory limits we will only take a random sample of the original test data

tst, tst_y = separate(camelyon['test'].shuffle(500).take(1000).map(preprocess))

# Setting single output model

model_1 = tf.keras.Sequential([
              tf.keras.Input(shape=(96, 96, 3)),
              layers.Conv2D(64, (7, 7), strides=2, padding='SAME', activation='relu',
                    kernel_initializer=tf.keras.initializers.glorot_normal,
                    bias_initializer=tf.keras.initializers.Constant(value=0.2)),
              layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="SAME"),
              layers.Conv2D(64, (1, 1), strides=1, padding='SAME', activation='relu',
                    kernel_initializer=tf.keras.initializers.glorot_normal,
                    bias_initializer=tf.keras.initializers.Constant(value=0.2)),
              layers.Conv2D(192, (3, 3), strides=1, padding='SAME', activation='relu',
                    kernel_initializer=tf.keras.initializers.glorot_normal,
                    bias_initializer=tf.keras.initializers.Constant(value=0.2)),
              layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="SAME"),
              Inception_st3(0),
              Inception_st3(1),
              layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="SAME"),
              Inception_st4(0),
              Inception_st4(1),
              Inception_st4(2),
              Inception_st4(3),
              Inception_st4(4),
              layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="SAME"),
              Inception_st5(0),
              Inception_st5(1),
              layers.GlobalAveragePooling2D(),
              layers.Dropout(0.4),
              layers.Dense(1, activation='sigmoid', name='main_output')
])

model_1.build()

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

model_1.compile(loss='binary_crossentropy',
                    optimizer=sgd, metrics=['binary_accuracy'])

model_1.fit(train, epochs=epochs, validation_data=val, callbacks=call_backs)

# save weights just in case

#import os

# model_1.save_weights("model_1.h5")

loss = tf.keras.losses.BinaryCrossentropy()

# Wrapping model into ART model
classifier = TensorFlowV2Classifier(model_1, nb_classes=1, loss_object=loss,
                                    input_shape=(96, 96, 3), clip_values=(0, 1))

from art.attacks.evasion import FastGradientMethod

# Crafting attack
attack = FastGradientMethod(classifier, norm=np.inf, eps=8, eps_step=8, targeted=False,
                            num_random_init=0, batch_size=1, minimal=False)

# Getting adversarial images
x_adv = attack.generate(tst)

# Evaluating model against all test data
model_1.evaluate(test)

# Evaluating one output model against sample from test data
model_1.evaluate(tst, tst_y)

# Evaluating model against adversarial images
model_1.evaluate(x_adv, tst_y)

# Evaluation proved the attack to be successful even though to a certain extent.
# Interestingly, as mentioned, the model without auxiliary outputs
# performs no much worse than the whole model.