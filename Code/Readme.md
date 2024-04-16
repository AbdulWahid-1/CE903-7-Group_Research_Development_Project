To Start with our GAN15 model, there are two options,

    1)- Jupyter Notebook (Main.ipynb)
    2)- Graphical User Interface (GUI.py)

The Functionality of both Jupyter Notebook (Main.ipynb) and Graphical User Interface (GUI.py) is same.

Both have save 2 Functions,

    1)- Train Model:

        train function has 7 argumnets:

        train(real_dataset, epochs, learning_rate, batch_size, save_model, model_name, path)

        As 1st argument you have to give your real unpreprocesed dataset.
        2nd, 3rd and 4th arguments are epochs, learning rate, batch size respectiviely.
        5th Argument is an optional argument, if set True then the function will save the model with the same name as you have specified in the 6th argument.
        6th argument is also an optional argumnet which is actually the name of the model you want to save.
        7th argument is the path where you want to save your model.
        When your 5th argumet is set True, the function will save 3 files:

            1)- The trained model. extension will be (.h5)
            2)- The min max saclling file named as (scalling.pkl)
            3)- The featues file, which contains the column names, and some unique features. named as (features.pkl)
        
    2)- Generate Samples:

        generator function has 4 arguments:

        generate_sample(trained_model, features_file, number_of _samples, saving_file_name, path)

        As first argument you have to give the name of trained model file.
        2nd is the number of samples you want to generate.
        3rd is the name of file with which you want to save the generated samples.
        4th is the path where your model is saved. The output will also be saved at that path.