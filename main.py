import cv2
import nltk
import numpy
import tensorflow as tf
from keras.layers import Conv2DTranspose, Dense, Input, Reshape
from keras.models import Model
from PIL import Image

from neural_network import NeuralNetwork
from window import Window


class GPUDevice:
    def __init__(self) -> None:
        pass

    def check_gpu_is_available(self):
        psychical_device = tf.config.list_physical_devices("GPU")
        print(psychical_device)
        # return [x.name for x in local_device_protos if x.device_type == 'GPU']


def main():
    input_dim = 100
    output_dim = 512 * 512 * 3  # Rozmiar obrazu
    epochs = 10000
    batch_size = 64
    hidden_units = 256
    gpu_device = GPUDevice()
    gpu_device.check_gpu_is_available()
    generator = NeuralNetwork(input_dim, output_dim, epochs, batch_size)
    new_window = Window()
    new_window.create_new_window()


if __name__ == "__main__":
    main()
