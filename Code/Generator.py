import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, ReLU, LeakyReLU, BatchNormalization

class Generator:
    def __init__(self, input_shape, output_shape, hidden_sizes):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_sizes = hidden_sizes
        self.model = self.build_model()

    def build_model(self):
        input_layer = Input(shape=self.input_shape)
        x = Dense(self.hidden_sizes[0])(input_layer)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        # x = ReLU()(x)
        # x = BatchNormalization()(x)
        for size in self.hidden_sizes[1:]:
            x = Dense(size)(x)
            x = LeakyReLU()(x)
            x = BatchNormalization()(x)
            # x = ReLU()(x)
            # x = BatchNormalization()(x)
        output_layer = Dense(self.output_shape[0], activation='sigmoid')(x)
        # model = Sequential(output_layer)

        model = Model(input_layer, output_layer)
        return model

    def generate_samples(self, n_samples):
        noise = np.random.normal(0, 1, size=(n_samples,) + self.input_shape)
        generated_samples = self.model.predict(noise)
        return generated_samples
    
    def save_model(self, model_name):
        self.model.save(model_name + '.h5')