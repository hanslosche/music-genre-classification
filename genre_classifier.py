import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

# path to json file that stores mfccs and genre labels for each processed segment
data_path = "genre_dataset/data.json"

def load_data(data_path):
    """loads training dataset from json file.

        :param data_path (str): path to json file containing data
        :return x (ndarray): inputs
        :return y (ndarray): targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    x = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("data succesfully loaded!")

    return  x, y


if __name__ == "__main__":

    # load data
    x, y = load_data(data_path)

    # create train/test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # build network topology
    model = keras.Sequential([

        # input layer
        keras.layers.Flatten(input_shape=(x.shape[1], x.shape[2])),

        # 1st dense layer
        keras.layers.Dense(512, activation='relu'),

        # 2nd dense layer
        keras.layers.Dense(256, activation='relu'),

        # 3rd dense layer
        keras.layers.Dense(64, activation='relu'),

        # output layer
        keras.layers.Dense(10, activation='softmax')
    ])

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train model
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=50)




