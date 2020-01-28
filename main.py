# Copyright 2019 Doyoung Gwak (tucan.dev@gmail.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ======================
#-*- coding: utf-8 -*-

import tensorflow as tf

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

import os
import pathlib

# relate link: https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK']='True'

print(tf.__version__)  # 2.0.0

from callbacks_model import get_check_pointer_callback
from callbacks_model import get_tensorboard_callback

from transferlearning_model import TransferLearningModel


# ======================================================================
# ======================================================================
# Configure dataset path

base_dataset_path = "/Volumes/tucan-SSD/datasets/coco/tucan9389_generated_dataset/generated_orientation_dataset"

train_dataset_path = os.path.join(base_dataset_path, "unlabeled2017_th")  # train dataset path
validation_dataset_path = os.path.join(base_dataset_path, "val2017_th")  # validation dataset path
test_dataset_path = os.path.join(base_dataset_path, "test2017_th")  # test dataset path

base_dataset_path = pathlib.Path(base_dataset_path)

train_dataset_path = pathlib.Path(train_dataset_path)
validation_dataset_path = pathlib.Path(validation_dataset_path)
test_dataset_path = pathlib.Path(test_dataset_path)

# check number of dataset images
train_image_count = len(list(train_dataset_path.glob('*/*.jpg')))
image_count = train_image_count
print("train_image_count:", train_image_count)

validation_image_count = len(list(validation_dataset_path.glob('*/*.jpg')))
print("validation_image_count:", validation_image_count)

test_image_count = len(list(test_dataset_path.glob('*/*.jpg')))
print("test_image_count:", test_image_count)

# number of classes
CLASS_NAMES = np.array([item.name for item in train_dataset_path.glob('*') if item.name != "LICENSE.txt"])
number_of_classes = len(CLASS_NAMES)
print("CLASS_NAMES:", CLASS_NAMES)

# show some image examples
images = list(train_dataset_path.glob('portrait/*'))

for image_path in images[:3]:
    im = Image.open(image_path)
    plt.imshow(im)
    plt.show()

# ======================================================================
# ======================================================================
# Make generator

# The 1./255 is to convert from uint8 to float32 in range [0,1].
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

train_generator = image_generator.flow_from_directory(
    train_dataset_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    classes=list(CLASS_NAMES),
    subset='training')

val_generator = image_generator.flow_from_directory(
    validation_dataset_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    classes=list(CLASS_NAMES),
    subset='validation')

def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        _ = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n])
        plt.title(CLASS_NAMES[label_batch[n] == 1][0].title())
        plt.axis('off')

# image_batch, label_batch = next(train_generator)
# # show_batch(image_batch, label_batch)

# ======================================================================
# ======================================================================
# setup output

current_time = datetime.now().strftime("%m%d%H%M")
output_path = "/Volumes/tucan-SSD/ml-project/orientation-detection/output"
if not os.path.exists(output_path):
    os.mkdir(output_path)

output_model_name = "_mobilenetv2"  # mobilenetv2
# output_base_model_name = "_{}".format(model_config.base_model_name)
# output_learning_rate = "_lr{}".format(train_config.learning_rate)
# output_decoder_filters = "_{}".format(model_config.filter_name)

output_name = current_time + output_model_name  # + output_learning_rate  # + output_decoder_filters

model_path = os.path.join(output_path, "models")
if not os.path.exists(model_path):
    os.mkdir(model_path)

log_path = os.path.join(output_path, "logs")
if not os.path.exists(log_path):
    os.mkdir(log_path)

print("\n")
print("model path:", model_path)
print("log path  :", log_path)
print("model name:", output_name)
print("\n")

# ======================================================================
# ======================================================================
# Setup output

# output model file(.hdf5)
check_pointer_callback = get_check_pointer_callback(model_path=model_path, output_name=output_name)

# output tensorboard log
tensorboard_callback = get_tensorboard_callback(log_path=log_path, output_name=output_name)

if __name__ == '__main__':

    # ======================================================================
    # ======================================================================
    # Build model

    IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

    model = TransferLearningModel(input_shape=IMG_SHAPE, number_of_classes=number_of_classes)

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    print('Number of trainable variables = {}'.format(len(model.trainable_variables)))

    # ======================================================================
    # ======================================================================
    # TRAINING

    epochs = 5

    history = model.fit(train_generator,
                        epochs=epochs,
                        validation_data=val_generator,
                        callbacks=[
                            check_pointer_callback,
                            tensorboard_callback]
                        )

    # ======================================================================
    # ======================================================================
    # Show result

    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']
    #
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    #
    # plt.figure(figsize=(8, 8))
    # plt.subplot(2, 1, 1)
    # plt.plot(acc, label='Training Accuracy')
    # plt.plot(val_acc, label='Validation Accuracy')
    # plt.legend(loc='lower right')
    # plt.ylabel('Accuracy')
    # plt.ylim([min(plt.ylim()), 1])
    # plt.title('Training and Validation Accuracy')
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(loss, label='Training Loss')
    # plt.plot(val_loss, label='Validation Loss')
    # plt.legend(loc='upper right')
    # plt.ylabel('Cross Entropy')
    # plt.ylim([0, 1.0])
    # plt.title('Training and Validation Loss')
    # plt.xlabel('epoch')
    # plt.show()

    # ======================================================================
    # ======================================================================
    # Fine-tuning phase

    model.configureForFinetuning()

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(1e-5),
                  metrics=['accuracy'])

    model.summary()

    print('Number of trainable variables = {}'.format(len(model.trainable_variables)))

    history_fine = model.fit(train_generator,
                             epochs=5,
                             validation_data=val_generator,
                             callbacks=[
                                 check_pointer_callback,
                                 tensorboard_callback]
                             )