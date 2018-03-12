

from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, GaussianNoise
from keras.models import Model
from keras import backend as K
from keras import regularizers
import os


class conv_encoder_model:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.depth=1
        self.autoencoder_model = None
        self.encoder_model = None
        self.decoder_model = None
        self.bottleneck_shape = None

        self.model_name = "##"
        self.model_to_load_from = "##"

        self.names  =[
            "convmodel_depth_0.h5",
            "convmodel_depth_1.h5",
            "convmodel_depth_2.h5",
            "convmodel_depth_3.h5",
            "convmodel_depth_4.h5"
        ]


    def nets(self, input_img):
        poollayer = 0

        part_1 = [
            [0, 'encode', 'c_1', 20],

            [1, 'encode', 'c_2', 20],
            [1, 'encode', 'pool', poollayer],

            [2, 'encode', 'c_3', 20],
            [2, 'encode', 'c_4', 20],
            [2, 'encode', 'pool', poollayer],

            [3, 'encode', 'c_5', 20],
            [3, 'encode', 'c_6', 20],
            [3, 'encode', 'pool', poollayer],

            [4, 'encode', 'c_7', 20],
            [4, 'encode', 'c_8', 20],
            [4, 'encode', 'pool', poollayer],

            [0, 'encode', 'c_z1', self.input_shape[2]],
         ]

        part_2= [
            [0, 'encode', 'c_z2', 20],

            [4, 'decode', 'c_-10', 20],
            [4, 'decode', 'pool', poollayer],
            [4, 'decode', 'c_-9', 20],

            [3, 'decode', 'c_-8', 20],
            [3, 'decode', 'pool', poollayer],
            [3, 'decode', 'c_-7', 20],

            [2, 'decode', 'c_-6', 20],
            [2, 'decode', 'pool', poollayer],
            [2, 'decode', 'c_-4', 20],

            [1, 'decode', 'c_-3', 20],
            [1, 'decode', 'pool', poollayer],
            [1, 'decode', 'c_-2', 20],

            [0, 'decode', 'c_-1', self.input_shape[2]],
        ]

        x = input_img

        x=self.partnet(part_1, x)
        encoded = x
        x = GaussianNoise(0.05, name='noise')(encoded)

        x = self.partnet(part_2, x)


        decoded = x

        return encoded, decoded

    def partnet(self, part, x):
        masksz = 5
        for n_layer in range(len(part)):
            layer_info = part[n_layer]
            if layer_info[0] > self.depth:
                continue
            name = layer_info[2]
            if name != 'pool':
                x = Conv2D(layer_info[3], (masksz, masksz), activation='relu', padding='same', name=name)(x)
            else:
                if layer_info[1] == 'decode':
                    x = UpSampling2D((2, 2))(x)
                else:
                    x = MaxPooling2D((2, 2))(x)
        return x

    def makenet(self):

        self.model_name = self.names[self.depth]
        input_img = Input(shape=self.input_shape)

        encoded, decoded = self.nets(input_img)

        self.autoencoder_model = Model(input_img, decoded)
        self.autoencoder_model.compile(optimizer='adam', loss='mean_squared_error')

        self.encoder_model = Model(input_img, encoded)

        bottleneck_layer=0
        for bottleneck_layer in range(len(self.autoencoder_model.layers)):
            if self.autoencoder_model.layers[-bottleneck_layer - 1].get_config()['name'] == 'noise':
                break

        bn = self.autoencoder_model.layers[-bottleneck_layer].input_shape
        self.bottleneck_shape = (bn[1], bn[2], bn[3])

        encoded_input = Input(shape=self.bottleneck_shape)

        dcdroutput = encoded_input
        for i in range (bottleneck_layer):
            i=i-bottleneck_layer
            dcdrlayer = self.autoencoder_model.layers[i]
            dcdroutput = dcdrlayer(dcdroutput)

        self.decoder_model = Model(encoded_input, dcdroutput)



        if os.path.isfile(self.model_to_load_from):
            try:
                #self.autoencoder_model.load_weights(self.model_name)
                self.autoencoder_model.load_weights(self.model_to_load_from, by_name=True)
                print("model loaded ", self.model_to_load_from)

            except ValueError:
                print("failed to load, new model shape?")
        else:print("new file ", self.model_to_load_from)



