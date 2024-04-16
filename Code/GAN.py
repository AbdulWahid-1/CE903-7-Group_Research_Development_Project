import numpy as np
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input

class GAN:
    def __init__(self, generator, discriminator, learning_rate):
        self.generator = generator
        self.discriminator = discriminator
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        self.discriminator.model.trainable = False
        gan_input = Input(shape=self.generator.input_shape)
        gan_output = self.discriminator.model(self.generator.model(gan_input))
        model = Model(gan_input, gan_output)
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.learning_rate))
        return model

    def train(self, real_data, epochs, batch_size):
        history = {"d_loss": [], "g_loss": []}

        for epoch in range(epochs):
            # Sample a batch of real data
            real_data_batch = real_data[np.random.choice(len(real_data), batch_size, replace=False)]

            # Train the discriminator on the real data
            discriminator_loss_real = self.discriminator.model.train_on_batch(real_data_batch, np.ones((batch_size, 1)))

            # Generate a batch of synthetic data using the generator
            synthetic_data_batch = self.generator.generate_samples(batch_size)

            # Train the discriminator on the synthetic data
            discriminator_loss_synthetic = self.discriminator.model.train_on_batch(synthetic_data_batch, np.zeros((batch_size, 1)))

            # Calculate the average discriminator loss
            discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_synthetic)

            # Train the generator to fool the discriminator
            generator_loss = self.model.train_on_batch(np.random.normal(0, 1, size=(batch_size,) + self.generator.input_shape), np.ones((batch_size, 1)))

            history["d_loss"].append(discriminator_loss)
            history["g_loss"].append(generator_loss)

            # Print the progress
            if epoch % 100 == 0:
                print(f'Epoch: {epoch}, Discriminator Loss: {discriminator_loss}, Generator Loss: {generator_loss}')
        return history
    
    def save_model(self, model_name):
        self.model.save(model_name + '.h5')