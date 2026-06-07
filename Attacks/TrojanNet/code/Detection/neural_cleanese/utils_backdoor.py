#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-05 11:30:01
# @Author  : Bolun Wang (bolunwang@cs.ucsb.edu)
# @Link    : http://cs.ucsb.edu/~bolunwang


import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


def dump_image(x, filename, format):
    img = image.array_to_img(x, scale=False)
    img.save(filename, format)
    return


def fix_gpu_memory(mem_fraction=1):
    """Configure GPU memory growth for TensorFlow 2.x"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if mem_fraction < 1:
                # Set memory limit if fraction is specified
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=int(mem_fraction * 8192))])  # Assuming 8GB GPU
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    return None


def load_dataset(data_filename, keys=None):
    ''' assume all datasets are numpy arrays '''
    dataset = {}
    with h5py.File(data_filename, 'r') as hf:
        if keys is None:
            for name in hf:
                dataset[name] = np.array(hf.get(name))
        else:
            for name in keys:
                dataset[name] = np.array(hf.get(name))

    return dataset
