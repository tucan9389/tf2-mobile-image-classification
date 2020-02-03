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

def build_model(input_shape, number_of_classes):
    # Create the base model from the pre-trained model MobileNet V2
    base_model = keras.applications.MobileNetV2(input_shape=input_shape,
                                                        include_top=False,
                                                        weights='imagenet')

    for layer in base_model.layers:
        layer.trainable = False

    model = keras.Sequential([
        base_model,
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(number_of_classes, activation='softmax')
    ])

    return base_model, model

def make_trainable(base_model=None):
    #base_model.trainable = True

    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))

    # Fine tune from this layer onwards
    fine_tune_at = 100

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False