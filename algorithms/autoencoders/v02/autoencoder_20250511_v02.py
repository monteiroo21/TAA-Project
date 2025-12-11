import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Model, load_model
from keras.datasets import mnist
import os
import math

keras.backend.clear_session() # Clear any previous session

log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

DATASET_BUFFER_SIZE = 100
DATASET_BATCH_SIZE = 1

IMG_WIDTH = 256
IMG_HEIGHT = 256

def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]

    w = w // 2
    input_image = image[:, :w, :]
    real_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image

# Function to resize an image to the desired dimensions
def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

# Function to normalize the pixel values in an image to the range [-1, 1]
def normalize(input_image, real_image):
    input_image = tf.image.rgb_to_grayscale(input_image)
    real_image = tf.image.rgb_to_grayscale(real_image)
    
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image

# Function to randomly jitter an image
def random_jitter(input_image, real_image):
    input_image, real_image = resize(input_image, real_image, 286, 286)

    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    if tf.random.uniform(()) > 0.5:
        cropped_image = tf.image.flip_left_right(cropped_image)

    return cropped_image[0], cropped_image[1]

# Function to load and preprocess an image
def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)    
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

# Function to load and preprocess an image for testing (without random jitter)
def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

# Load the dataset    
train_dataset = tf.data.Dataset.list_files('./data/TCIRRP/train1k/*.jpg')
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(DATASET_BUFFER_SIZE)
train_dataset = train_dataset.batch(DATASET_BATCH_SIZE, drop_remainder=True)

test_dataset = tf.data.Dataset.list_files('./data/TCIRRP/test0.1k/*.jpg')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(DATASET_BATCH_SIZE, drop_remainder=True)

LATENT_DIMENSIONS = 64
    
class Autoencoder(Model):
    def __init__(self, **kwargs):
        super(Autoencoder, self).__init__(**kwargs)
        
        # Apply convulutional layers and move from flatten to latent hidden layer
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1)), # one channel (intensity)
            layers.Conv2D(32, 3, strides=2, padding='same', activation='relu'),
            layers.Conv2D(64, 3, strides=2, padding='same', activation='relu'),
            layers.Conv2D(128, 3, strides=2, padding='same', activation='relu'),
            layers.Flatten(),
            layers.Dense(LATENT_DIMENSIONS, activation='relu'),
        ])
        
        self.bw = math.ceil(IMG_WIDTH / 2**3) # width bottleneck
        self.bh = math.ceil(IMG_HEIGHT / 2**3) # height bottleneck
        self.bc = 128                 # channels bottleneck
                    
        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(LATENT_DIMENSIONS,)),
            layers.Dense(self.bw * self.bh * self.bc, activation='relu'),
            layers.Reshape((self.bw, self.bh, self.bc)),
            layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(1,  3, strides=2, padding='same', activation='relu'),
        ])
    
    def call(self, input_data):
        encoded = self.encoder(input_data)
        decoded = self.decoder(encoded)
        return decoded


## Model instantiation

input_model_path = "1k-50-epochs.keras"
output_model_path = "1k-100-epochs.keras"

if 1 and os.path.exists(input_model_path):
    print("\033[92mModel loaded. With evaluation:\033[0m")
    autoencoder = load_model(input_model_path, custom_objects={'Autoencoder': Autoencoder})
    autoencoder.evaluate(test_dataset, verbose=2)
else:
    print("Model not loaded")
    autoencoder = Autoencoder()
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    
autoencoder.summary()



## Model training

save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=output_model_path, verbose=2, save_best_only=True, monitor='val_loss')

tensorboard_cb = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,      # grava histogramas de pesos a cada época
    write_graph=True,      # salva a definição do grafo
    write_images=False     # imagens de pesos — não necessário aqui
)

autoencoder.fit(
    train_dataset,
    epochs=50,
    validation_data=test_dataset,
    callbacks=[save_callback, tensorboard_cb],
)

print("Final evaluation:")
autoencoder.evaluate(test_dataset, verbose=2)


## Model evaluation

print("\n\n\n")

print("Saved model evaluation:")
autoencoder = load_model(output_model_path, custom_objects={'Autoencoder': Autoencoder})
autoencoder.evaluate(test_dataset, verbose=2)

num_tests = 5

for name, dataset in [("Training set", train_dataset), ("Testing set", test_dataset)]:
    # 1) Extrair algumas amostras
    x_input = []
    x_real  = []
    for input_batch, real_batch in dataset.take(num_tests):
        # Assumindo batch_size = 1
        x_input.extend(input_batch)
        x_real.extend(real_batch)
    # empilha em tensores [num_tests * batch_size, ...]
    x_input = tf.stack(x_input)
    x_real  = tf.stack(x_real)

    # 2) Codifica e decodifica
    encoded_imgs = autoencoder.encoder(x_input).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    # 3) Plota
    plt.figure(figsize=(6, num_tests * 2))
    plt.suptitle(name, fontsize=16)
    for i in range(num_tests):
        # Coluna 1: original de entrada
        ax = plt.subplot(num_tests, 3, i*3 + 1)
        plt.imshow(x_input[i].numpy(), cmap="gray")
        ax.set_title("Input")
        ax.axis("off")

        # Coluna 2: reconstruída
        ax = plt.subplot(num_tests, 3, i*3 + 2)
        plt.imshow(decoded_imgs[i], cmap="gray")
        ax.set_title("Decoded")
        ax.axis("off")

        # Coluna 3: imagem “real” alvo (se for diferente da entrada)
        ax = plt.subplot(num_tests, 3, i*3 + 3)
        plt.imshow(x_real[i].numpy(), cmap="gray")
        ax.set_title("Target")
        ax.axis("off")
                

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
