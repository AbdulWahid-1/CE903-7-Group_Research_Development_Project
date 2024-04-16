from keras.models import Model
from keras.layers import Input, Dense, LeakyReLU
from keras.optimizers import Adam

class Discriminator:
    def __init__(self, input_shape, hidden_sizes):
        self.input_shape = input_shape
        self.hidden_sizes = hidden_sizes
        self.model = self.build_model()
        self.model.compile(loss='binary_crossentropy', optimizer=Adam())

    def build_model(self):
        input_layer = Input(shape=self.input_shape)
        x = Dense(self.hidden_sizes[0])(input_layer)
        x = LeakyReLU()(x)
        for size in self.hidden_sizes[1:]:
            x = Dense(size)(x)
            x = LeakyReLU()(x)
        output_layer = Dense(1, activation='sigmoid')(x)
        model = Model(input_layer, output_layer)
        return model
    
    def save_model(self, model_name):
        self.model.save(model_name + '.h5')