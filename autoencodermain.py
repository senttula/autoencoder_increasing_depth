from scipy.misc import imresize, toimage
from keras import backend as K
from keras.callbacks import Callback
import numpy as np

import conv_encoder
import os


dir_path = os.path.dirname(os.path.realpath(__file__))

orig_input_shape = (576, 192, 3)
save_folder = dir_path+'\\progress\\'
data_name = 'xdata.npy'

epochs = 10
patience  = 10

class LossHistory(Callback):
    def __init__(self, encoder):
        a=1
        self.encoder = encoder

    def on_train_begin(self, logs={}):
        self.losses = [[],[]]


    def on_epoch_end(self, epoch, logs={}):
        self.losses[0].append(logs.get('val_loss'))
        self.losses[1].append(logs.get('loss'))
        self.encoder.update_plot("loss "+str(min(self.losses[0])))

class encoder:
    def __init__(self):
        self.rs = 1
        self.input_shape = orig_input_shape
        self.epochn = 0
        self.epochs = epochs
        self.patience = patience

        self.losses = [999999,999999]

    def get_models(self):
        self.models = conv_encoder.conv_encoder_model(self.input_shape)

    def update_models(self, depth):
        K.clear_session()
        self.rs = 2**depth
        self.models.depth=depth
        self.models.makenet()
        self.autoencoder_model = self.models.autoencoder_model
        self.encoder_model = self.models.encoder_model
        self.decoder_model = self.models.decoder_model
        self.shape_encoded = self.models.bottleneck_shape

        print(self.autoencoder_model.summary())
        print("compression: ", round(np.prod(self.input_shape) / np.prod(self.shape_encoded), 2))
        print('compression shape ', self.shape_encoded)

    def load_data(self):

        data = np.load(data_name)
        split = int(data.shape[0] * 0.95)
        data = data.astype('float32') / 255.

        if 0:
            self.scaled_shape = (int(orig_input_shape[0] / self.rs), int(orig_input_shape[1] / self.rs), 3)
            self.input_shape = self.scaled_shape
            new = []
            for i in range (data.shape[0]):
                img = imresize(data[i], (self.scaled_shape[0], self.scaled_shape[1]))
                new.append(img)
            data = np.array(new)

        self.x_train, self.x_test = data[0:split, :, :, :], data[split:data.shape[0], :, :, :]

    def get_encodes(self):


        random_encoded = np.random.rand(self.shape_encoded[0], self.shape_encoded[1], self.shape_encoded[2])
        encoded_imgs = self.encoder_model.predict(self.x_test[0:4, :, :, :])
        encoded_train_data = self.encoder_model.predict(self.x_train)

        tester_encodes = []
        tester_encodes.append(np.median(encoded_imgs, axis=0))
        tester_encodes.append(np.median(encoded_train_data, axis=0))
        tester_encodes.append(np.add(random_encoded, np.mean(encoded_train_data, axis=0) * 3) / 4)
        tester_encodes.append(np.square(np.square(random_encoded)))

        noise = np.random.normal(0, 0.15, (encoded_imgs[0].shape))
        tester_encodes.append(np.add(noise, encoded_imgs[0]))

        tester_encodes = np.array(tester_encodes)

        return tester_encodes, encoded_imgs

    def update_plot(self, info=""):
        self.epochn += 1
        print("epoch: ",self.epochn, info)

        tester_encodes, encoded_imgs = self.get_encodes()

        tester_decoded = self.decoder_model.predict(tester_encodes)
        decoded_imgs = self.decoder_model.predict(encoded_imgs)

        for i in range(tester_decoded.shape[0]):
            data = tester_decoded[i]
            data = (data * 255).astype('int8')
            data = np.concatenate((data, reshape_encode(tester_encodes[i])), axis=1)
            save(data, i, self.epochn)

        for i in range(tester_decoded.shape[0], 9):
            index = i - tester_decoded.shape[0]
            data = decoded_imgs[index]
            data = (data * 255).astype('int8')
            data = np.concatenate((data, reshape_encode(encoded_imgs[index])), axis=1)
            save(data, i, self.epochn)

    def train(self):
        print("running...")

        if epochs == 0:
            self.update_plot()
            return
        self.losses = [99999, 99999]

        self.update_plot()
        callbacks = LossHistory(self)
        for epochcncnnc in range(self.epochs):
            print("training...")
            self.autoencoder_model.fit(self.x_train, self.x_train,
                                  epochs=self.patience,
                                  batch_size=int(4+32/self.rs),
                                  shuffle=True,
                                  callbacks=[callbacks],
                                  verbose=0,
                                  validation_data=(self.x_test, self.x_test))
            self.autoencoder_model.save_weights(self.models.model_name)
            print("saved, loss: ", min(callbacks.losses[0]),min(callbacks.losses[1]), self.losses)

            flag_stop = True
            if self.losses[0]>min(callbacks.losses[0]):
                self.losses[0] = min(callbacks.losses[0])
                flag_stop = False
            if self.losses[1]>min(callbacks.losses[1]):
                self.losses[1] = min(callbacks.losses[1])
                flag_stop = False

            if flag_stop:
                print("##################")
                print("stagnant, stopping")
                print("##################")
                break

    def do(self):
        self.load_data()
        self.get_models()

        self.models.model_to_load_from = "convmodel_depth_0.h5" # if exists
        self.update_models(0)
        self.train()

        self.models.model_to_load_from = "convmodel_depth_0.h5"
        self.update_models(1)
        self.train()

        self.models.model_to_load_from = "convmodel_depth_1.h5"
        self.update_models(2)
        self.train()

        self.models.model_to_load_from = "convmodel_depth_3.h5"
        self.update_models(3)
        self.train()

        self.models.model_to_load_from = "convmodel_depth_3.h5"
        self.update_models(4)
        self.train()


def reshape_encode(arr):
    arr = imresize(arr, orig_input_shape)
    return arr


def save(data, x, epochnumber):
    im = toimage(data)
    filename = "%s%s_%s_image.jpg" % (save_folder, str(x), str(epochnumber))
    im.save(filename)

if __name__ == '__main__':
    enc=encoder()
    enc.do()
