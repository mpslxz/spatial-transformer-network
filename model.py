from stn import spatial_transformer_network as transformer
from tensorflow.keras import layers, Model

class STModel(object):
    
    def __init__(self, input_shape):
        self.inpt = layers.Input(input_shape)
        self.output = self.transformer_net(self.inpt, self.localization_net(self.inpt))
        return Model(self.inpt, self.output)

    def localization_net(self, x):
        x = layers.Conv2D(10, (2,2), padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPool2D((2,2))(x)
        x = layers.Conv2D(20, (2,2), padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPool2D((2,2))(x)
        x = layers.Conv2D(30, (2,2), padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPool2D((2,2))(x)
        x = tf.flatten(x)
        x = layers.Dense(6, 25)(x)
        return x

    def transformer_net(self, x, theta):
        return transformer(x, theta)
