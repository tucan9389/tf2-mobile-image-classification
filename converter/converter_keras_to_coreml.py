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

import coremltools
import keras
import os

model_directory_path = "/Volumes/tucan-SSD/ml-project/orientation-detection/output/models"
# model_name = "02010020_mobilenetv2_final"
model_name = "02021422_mobilenetv2_final"

input_keras_model_path = os.path.join(model_directory_path, model_name + ".h5")
print("input_keras_model_path:", input_keras_model_path)

output_model_name = "" + model_name
output_tflite_model_directory_path = os.path.join(model_directory_path, "mlmodel")
if not os.path.exists(output_tflite_model_directory_path):
    os.mkdir(output_tflite_model_directory_path)
output_mlmodel_model_path = os.path.join(output_tflite_model_directory_path, "orientation_detection_" + output_model_name + ".mlmodel")
print("output_mlmodel_model_path:", output_mlmodel_model_path)

IMG_HEIGHT = 224
IMG_WIDTH = 224

if __name__ == '__main__':

    # Load model
    print("Load model")
    keras_model = keras.models.load_model(input_keras_model_path)

    # get input, output node names for the TF graph from the Keras model
    input_name = keras_model.inputs[0].name.split(':')[0]
    keras_output_node_name = keras_model.outputs[0].name.split(':')[0]
    graph_output_node_name = keras_output_node_name.split('/')[-1]

    print(keras_model.inputs)
    print("input_name:", input_name)
    print("keras_output_node_name:", keras_output_node_name)
    print("graph_output_node_name:", graph_output_node_name)

    # mlmodel = coremltools.converters.keras.convert(keras_model,
    #                                                # input_names='image',
    #                                                image_input_names='image',
    #                                                image_scale=1./255.,
    #                                                class_labels=['landscape-left' 'landscape-right' 'portrait' 'upside-down'])
    mlmodel = coremltools.converters.keras.convert(keras_model,
                                                   image_input_names='image',
                                                   image_scale=1./255.,
                                                   use_float_arraytype=True)

    # # convert this model to Core ML format
    # mlmodel = tfcoreml.convert(tf_model_path=input_keras_model_path,
    #                            input_name_shape_dict={input_name: (1, IMG_HEIGHT, IMG_WIDTH, 3)},
    #                            image_input_names=[input_name],
    #                            # is_bgr=False,
    #                            # red_bias=-1.0,
    #                            # green_bias=-1.0,
    #                            # blue_bias=-1.0,
    #                            image_scale=1./255,
    #                            output_feature_names=[graph_output_node_name],
    #                            # class_labels=['none', 'landscape-left' 'landscape-right' 'portrait' 'upside-down'],
    #                            class_labels=['landscape-left' 'landscape-right' 'portrait' 'upside-down'],
    #                            minimum_ios_deployment_target='13')

    mlmodel.author = 'Doyoung Gwak'
    mlmodel.license = 'Apache License 2.0'

    # print(mlmodel.input_descriptions)
    # print(mlmodel.output_descriptions)

    # print(mlmodel.get_spec().description)  # "mobilenetv2_1.00_224_input"
    # spec = mlmodel.get_spec()
    # spec.description.input[0].name = "image"
    # spec.description.input[0].shortDescription = "Input image"
    # spec.description.output[0].name = "labels"
    # spec.description.output[0].shortDescription = "Predicted class confidence"
    #convert_multiarray_output_to_image(spec, 'imageOutput', is_bgr=True)
    # mlmodel = coremltools.models.MLModel(spec)

    print(mlmodel.get_spec().description)
    mlmodel.save(output_mlmodel_model_path)