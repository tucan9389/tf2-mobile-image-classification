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

import keras
# from keras.applications.resnet50 import ResNet50
# from keras.preprocessing import image
# from keras.applications.resnet50 import preprocess_input, decode_predictions
# import numpy as np
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

import os
import pathlib

from common.utils import remove_all_DS_STORE_reculsively

from trainer.keras.transferlearning_model import build_model
from trainer.keras.transferlearning_model import make_trainable
from trainer.keras.callbacks_model import get_check_pointer_callback
from trainer.keras.callbacks_model import get_tensorboard_callback



# ======================================================================
# ======================================================================
# Configure dataset path

# config
# base_dataset_path = "/Volumes/tucan-SSD/datasets/coco/tucan9389_generated_dataset/generated_orientation_dataset_4class_224x224"
base_dataset_path = "/Volumes/tucan-SSD/datasets/coco/tucan9389_generated_dataset/tmp_dataset_4class_224x224"
remove_all_DS_STORE_reculsively(path=base_dataset_path)

train_dataset_path = os.path.join(base_dataset_path, "unlabeled2017_224x224")  # train dataset path
validation_dataset_path = os.path.join(base_dataset_path, "val2017_224x224")  # validation dataset path
test_dataset_path = os.path.join(base_dataset_path, "test2017_224x224")  # test dataset path

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
CLASS_NAMES = np.array([item.name for item in train_dataset_path.glob('*') if item.name != "LICENSE.txt" and item.name != ".DS_Store"] )
number_of_classes = len(CLASS_NAMES)
print("CLASS_NAMES:", CLASS_NAMES)

# show some image examples
images = list(train_dataset_path.glob('portrait/*'))

# for image_path in images[:3]:
#     im = Image.open(image_path)
#     plt.imshow(im)
#     plt.show()

# ======================================================================
# ======================================================================
# Make generator

# The 1./255 is to convert from uint8 to float32 in range [0,1].
image_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

BATCH_SIZE = 32  # hyper parameter
IMG_HEIGHT = 224  # hyper parameter
IMG_WIDTH = 224  # hyper parameter
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

train_generator = image_generator.flow_from_directory(
    train_dataset_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    classes=list(CLASS_NAMES),
    subset='training'
)

val_generator = image_generator.flow_from_directory(
    validation_dataset_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    classes=list(CLASS_NAMES),
    # subset='validation'
)

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

# config
output_model_name = "_mobilenetv2"  # mobilenetv2
# output_base_model_name = "_{}".format(model_config.base_model_name)
# output_learning_rate = "_lr{}".format(train_config.learning_rate)
# output_decoder_filters = "_{}".format(model_config.filter_name)

output_name = current_time + output_model_name  # + output_learning_rate  # + output_decoder_filters

# config
model_path = os.path.join(output_path, "models")
if not os.path.exists(model_path):
    os.mkdir(model_path)

# config
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

# feature extraction model
feature_extraction_output_name = output_name + "_feature-extraction"
# output model file(.h5)
check_pointer_callback_fe = get_check_pointer_callback(model_path=model_path, output_name=feature_extraction_output_name)
# output tensorboard log
tensorboard_callback_fe = get_tensorboard_callback(log_path=log_path, output_name=feature_extraction_output_name)

# fine-tuning model
fine_tuning_output_name = output_name + "_fine-tuning"
# output model file(.h5)
check_pointer_callback_ft = get_check_pointer_callback(model_path=model_path, output_name=fine_tuning_output_name)
# output tensorboard log
tensorboard_callback_ft = get_tensorboard_callback(log_path=log_path, output_name=fine_tuning_output_name)

IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)  # config

if __name__ == '__main__':

    # ======================================================================
    # ======================================================================
    # Build model

    base_model, model = build_model(input_shape=IMG_SHAPE, number_of_classes=number_of_classes)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    #print('Number of trainable variables = {}'.format(len(model.trainable_variables)))

    # ======================================================================
    # ======================================================================

    feature_extraction_epochs = 1
    fine_tuning_epochs = 1

    # ======================================================================
    # ======================================================================
    # TRAINING1 - feature extraction step

    history = model.fit_generator(train_generator,
                                  epochs=feature_extraction_epochs,
                                  steps_per_epoch=STEPS_PER_EPOCH,
                                  validation_data=val_generator,
                                  validation_steps=32,
                                  callbacks=[
                                      check_pointer_callback_fe,
                                      tensorboard_callback_fe]
                                  )

    # config
    output_model_path = os.path.join(model_path, output_name + "_middle" + ".h5")
    print("save to '", output_model_path, "'")
    model.save(output_model_path)

    # ======================================================================
    # ======================================================================
    # TRAINING2 - fine tuning step

    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))

    # Fine tune from this layer onwards
    make_trainable(base_model=base_model)

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(1e-5),
                  metrics=['accuracy'])

    model.summary()

    #print('Number of trainable variables = {}'.format(len(model.trainable_variables)))

    history_fine = model.fit_generator(train_generator,
                                       epochs=fine_tuning_epochs,
                                       steps_per_epoch=STEPS_PER_EPOCH,
                                       validation_data=val_generator,
                                       validation_steps=32,
                                       callbacks=[
                                           check_pointer_callback_ft,
                                           tensorboard_callback_ft]
                                       )

    # config
    output_model_path = os.path.join(model_path, output_name + "_final" + ".h5")
    print("save to '", output_model_path, "'")
    model.save(output_model_path)