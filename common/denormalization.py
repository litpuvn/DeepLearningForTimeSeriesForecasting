from keras import backend as K
from keras.layers import Layer
from keras import initializers


class Denormalization(Layer):
    def __init__(self, **kwargs):
        super(Denormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Denormalization, self).build(input_shape)

    def call(self, x):
        return x[0] * x[1][:, :, 0] * x[1][:, :, 1]

    def compute_output_shape(self, input_shape):
        return input_shape[0]