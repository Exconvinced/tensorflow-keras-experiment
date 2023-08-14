import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    # Initialize lists to store images and labels:
    images, labels = list(), list()

    # Define the dimension to resize images to:
    resize_dim = (IMG_WIDTH, IMG_HEIGHT)

    # Loop over all category folders inside data_dir:
    for folder in os.listdir(data_dir):
        path = os.path.join(data_dir, folder)

        # Loop over all image files inside each category folder:
        for file in os.listdir(path):
            image = os.path.join(path, file)

            # Read image file and resize to IMG_WIDTH x IMG_HEIGHT:
            image = cv2.imread(image)
            image = cv2.resize(image, resize_dim)

            # Append image and label to lists
            images.append(image)
            labels.append(int(folder))

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Try to use GPU if available:
    with tf.device('/device:GPU:0'):
        # Let tensorflow.keras be k:
        k = tf.keras

        # Define the input shape:
        input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)

        # Initialize a sequential model and build as we stack layers:
        model = k.models.Sequential()

        # Add definition of input shape:
        model.add(k.Input(shape=input_shape))

        # Add two convolutional layers:
        filters = 32 # Higher number of filters = more features but more complexity
        kernel_size = (3, 3) # Smaller kernel size = detects finer details and edges
        activation = "relu" # RELU is the default activation function
        model.add(k.layers.Conv2D(filters, kernel_size, activation=activation))
        model.add(k.layers.Conv2D(filters, kernel_size, activation=activation))

        # Add max-pooling layer:
        pool_size = (3, 3) # Higher size = lower resolution but captures more features
        model.add(k.layers.MaxPooling2D(pool_size=pool_size))
        
        # Apply flatten layer:
        model.add(k.layers.Flatten())

        # Add two hidden layers:
        units = 128 # Higher number of units = more complex model
        model.add(k.layers.Dense(units, activation=activation))
        model.add(k.layers.Dense(units, activation=activation))

        # Add dropout layer:
        # rate = 0.5 # Higher rate = more dropout
        # model.add(k.layers.Dropout(rate))

        # Finally, add output layer:
        units = NUM_CATEGORIES # Number of categories
        activation = "softmax" # Softmax is the default activation function
        model.add(k.layers.Dense(units, activation=activation))

        # Compile model:
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model


if __name__ == "__main__":
    main()
