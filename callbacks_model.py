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

import os
import tensorflow as tf

def get_check_pointer_callback(model_path, output_name):
    checkpoint_path = os.path.join(model_path, output_name + ".hdf5")  # ".ckpt"
    check_pointer_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                                save_weights_only=False,
                                                                verbose=1)
    return check_pointer_callback


def get_tensorboard_callback(log_path, output_name):
    log_path = os.path.join(log_path, output_name)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=True,
                                                          write_images=True)

    return tensorboard_callback