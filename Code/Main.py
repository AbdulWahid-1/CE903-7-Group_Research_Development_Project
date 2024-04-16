from Preprocessing import Preprocess
from Post_processing import Post_processing
from Generator import Generator
from Discriminator import Discriminator
from GAN import GAN
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from pickle import dump
import pandas as pd 
import os

def train(data, epochs, learning_rate, batch_size, save_model=True, model_name='generator', path='new_model'):
    n_feature = len(data.columns)
    preprocesss = Preprocess(data)
    dataset_real = preprocesss.preprocessor()
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset_real)
    hidden_sizes = [50, 120, 200, 120, 50]
    generator = Generator((n_feature,), (n_feature,), hidden_sizes)
    discriminator = Discriminator((n_feature,), hidden_sizes)

    # Create the GAN model
    gan = GAN(generator, discriminator, learning_rate)
    history = gan.train(dataset, epochs, batch_size)
    if save_model == True:
        generator.save_model(path + '/' +model_name)
        preprocesss.save_features(path)
        dump(scaler, open(path + '/scalling.pkl', 'wb'))
    
    model_output_samples = generator.generate_samples(1000)
    synthetic_data = scaler.inverse_transform(model_output_samples)
    synthetic_data = synthetic_data.round(decimals=0)
    synthetic_data = pd.DataFrame(synthetic_data, columns=dataset_real.columns)

    return history, dataset_real, synthetic_data

def generate_samples(trained_model, n_samples, generated_file_name, path='new_model'):
    output_shape = 48
    path = os.path.dirname(os.path.abspath(trained_model))
    print(path)
    my_model = tf.keras.models.load_model(trained_model)
    noise = np.random.normal(0, 1, size=(n_samples,) + (output_shape,))
    model_output_samples = my_model.predict(noise)
    postp = Post_processing(model_output_samples, path)
    generated_samples = postp.process(generated_file_name, path)
    return generated_samples