import cv2
import nltk
import numpy
import tensorflow as tf
from keras.layers import Conv2DTranspose, Dense, Input, Reshape
from keras.models import Model
from PIL import Image


class NeuralNetwork:
    def __init__(self, input_dim, output_dim, epochs, batch_size):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epochs = epochs
        self.batch_size = batch_size

        self.build_generator_model()

    def build_generator_model(self):
        generator_input = Input(shape=(self.input_dim,))
        x = Dense(256, activation="relu")(generator_input)
        x = Dense(128 * 128 * 256, activation="relu")(x)
        x = Reshape((128, 128, 256))(x)
        x = Conv2DTranspose(
            64, kernel_size=4, strides=2, padding="same", activation="relu"
        )(x)
        generator_output = Conv2DTranspose(
            3, kernel_size=4, strides=2, padding="same", activation="sigmoid"
        )(x)

        generator_model = Model(generator_input, generator_output)
        generator_model.compile(loss="binary_crossentropy", optimizer="adam")
        self.train_network(generator_model)

    def generate_pattern(self, image_size):
        color_img = numpy.zeros((image_size, image_size, 3), dtype=numpy.float32)
        num_shapes = numpy.random.randint(10, 60)
        for _ in range(num_shapes):
            dot_color = (255, 128, 128)  # Różowy kolor (RGB)
            dot_size = numpy.random.randint(420, 480)  # Losowy rozmiar kropki
            x_pos = numpy.random.randint(0, image_size - dot_size)
            y_pos = numpy.random.randint(0, image_size - dot_size)
            cv2.circle(color_img, (x_pos, y_pos), dot_size, dot_color, -1)
            colored_img_flatten = color_img.flatten()
            color_img_resize = cv2.resize(colored_img_flatten, (1, 100))
        return color_img_resize

    def train_network(self, generator_model):
        training_data = numpy.random.randn(10000, self.input_dim)
        for epoch in range(self.epochs):
            batch_indices = numpy.random.randint(
                0, training_data.shape[0], self.batch_size
            )
            batch = training_data[batch_indices]

            generated_images = generator_model.predict(
                numpy.random.randn(self.batch_size, self.input_dim)
            )

            generator_loss = generator_model.train_on_batch(
                numpy.random.randn(self.batch_size, self.input_dim), generated_images
            )

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Generator Loss: {generator_loss}")
                # for i, image_array in enumerate(generated_images):
                # image = Image.fromarray((image_array * 255).astype(numpy.uint8))
                colored_img = self.generate_pattern(512)
                generated_pattern = generator_model.predict(
                    colored_img.reshape(1, self.input_dim)
                )

                image = Image.fromarray(
                    (generated_pattern[0] * 255).astype(numpy.uint8)
                )
                image.save(f"test_image/test_image_{epoch}.jpg")
