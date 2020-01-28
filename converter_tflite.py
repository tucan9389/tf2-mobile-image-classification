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
import os

model_directory_path = "/Volumes/tucan-SSD/ml-project/orientation-detection/output/models"
model_name = "01272307_mobilenetv2"

input_keras_model_path = os.path.join(model_directory_path, model_name + ".hdf5")

output_model_name = "" + model_name
output_tflite_model_directory_path = os.path.join(model_directory_path, "tflite")
if not os.path.exists(output_tflite_model_directory_path):
    os.mkdir(output_tflite_model_directory_path)
output_tflite_model_path = os.path.join(output_tflite_model_directory_path, "orientation_detection_" + output_model_name + ".tflite")

if __name__ == '__main__':
    # ======================================================================
    # ======================================================================
    # Load model
    print("Load model")

    model = tf.keras.models.load_model(input_keras_model_path)

    # ======================================================================
    # ======================================================================
    # Convert to tflite model
    print("Convert to tflite model")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(output_tflite_model_path, 'wb') as f:
        f.write(tflite_model)