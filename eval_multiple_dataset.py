import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Model, load_model
from keras.datasets import mnist
from tensorflow.data import AUTOTUNE
import os
import math
from itertools import product
import random
import scienceplots

plt.style.use('science')

keras.backend.clear_session() # Clear any previous session

DATASET_BUFFER_SIZE = 1000
DATASET_BATCH_SIZE = 32

IMG_WIDTH = 256
IMG_HEIGHT = 256

splits = ["split_1", "split_2"]
subsets: list[str] = ["subset_1", "subset_2", "subset_3", "subset_4"]

product = product(splits, subsets)


# 1) Camadas de augmentation específicas
data_augment = tf.keras.Sequential([
    # rotações entre 0 e 360 graus
    layers.RandomRotation(factor=1.0, fill_mode='reflect'),
    # zoom in/out leve (±10%)
    layers.RandomZoom(0.1, 0.1, fill_mode='reflect'),
    # contrate variações leves
    layers.RandomContrast(0.05),
    # brilho leve
    layers.RandomBrightness(0.05),
])

def preprocess_pair(path, training):
    # a) carregar e split
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=1)  # já grayscale
    w = tf.shape(img)[1] // 2
    # “real” é a metade esquerda, “inp” a direita
    real = img[:, :w, :]
    inp  = img[:, w:, :]

    # b) augment só em treino
    if training:
        # concatenamos para aplicar transform igual em ambos
        pair = tf.concat([inp, real], axis=2)  # shape (H, W, 2)
        pair = data_augment(pair)
        inp, real = tf.split(pair, num_or_size_splits=2, axis=2)

    # c) resize com bilinear + padding reflexivo p/ evitar artefatos
    inp  = tf.image.resize(inp,  [IMG_WIDTH, IMG_HEIGHT], method='bilinear')
    real = tf.image.resize(real, [IMG_WIDTH, IMG_HEIGHT], method='bilinear')

    # d) z-score normalization por imagem
    inp_mean, inp_var = tf.nn.moments(inp, axes=[0,1])
    real_mean, real_var = tf.nn.moments(real, axes=[0,1])
    inp  = (inp  - inp_mean)  / tf.sqrt(inp_var  + 1e-6)
    real = (real - real_mean) / tf.sqrt(real_var + 1e-6)

    return inp, real

def make_ds(pattern, training):
    ds = tf.data.Dataset.list_files(pattern)
    if training:
        ds = ds.shuffle(DATASET_BUFFER_SIZE)
    ds = ds.map(lambda p: preprocess_pair(p, training),
                num_parallel_calls=AUTOTUNE)
    ds = ds.batch(DATASET_BATCH_SIZE, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)
    return ds

######################################################################

LATENT_DIMENSIONS = 32
    
class Autoencoder(Model):
    def __init__(self, **kwargs):
        super(Autoencoder, self).__init__(**kwargs)
        
        # Apply convulutional layers and move from flatten to latent hidden layer
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1)), # one channel (intensity)
            
            # 1ª camada conv
            layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            
            # 2ª camada conv
            layers.Conv2D(64, 3, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            
            # 3ª camada conv
            layers.Conv2D(128, 3, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('relu'),
    
            layers.Flatten(),
            layers.Dense(LATENT_DIMENSIONS, activation='relu'),
        ])
        
        self.bw = math.ceil(IMG_WIDTH / 2**3)   # width bottleneck
        self.bh = math.ceil(IMG_HEIGHT / 2**3)  # height bottleneck
        self.bc = 128                           # channels bottleneck
                    
        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(LATENT_DIMENSIONS,)),
            
            layers.Dense(self.bw * self.bh * self.bc, use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            
            layers.Reshape((self.bw, self.bh, self.bc)),
            
            layers.Conv2DTranspose(64, 3, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            
            layers.Conv2DTranspose(32, 3, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            
            layers.Conv2DTranspose(1, 3, strides=2, padding='same'),
        ])
    
    def call(self, input_data):
        encoded = self.encoder(input_data)
        decoded = self.decoder(encoded)
        return decoded


sample_losses = []

for split, subset in product:

    file_with_names = f"./data/{split}/test/{subset}.txt"
    
    with open(file_with_names, 'r') as f:
        file_names = f.read().splitlines()
        
    print("Number of files:", len(file_names))
    
    test_dataset = make_ds(file_names, training=False)
    

    ## Model instantiation
    input_model_path = f"{split}-{subset}-30Epochs-32Lat-64Batch-linear.keras"

    print("\033[92mModel loaded. With evaluation:\033[0m")
    autoencoder = load_model(input_model_path, custom_objects={'Autoencoder': Autoencoder})
    # Ensure the dataset is repeated to avoid running out of data during evaluation

    # Evaluate the model on the test dataset
    autoencoder.evaluate(test_dataset, verbose=2)
        

    import numpy as np
    import matplotlib.pyplot as plt

    # 1) Reúna todas as imagens e reconstruções
    y_true = []
    y_pred = []

    for x_batch, y_batch in test_dataset:
        recon = autoencoder.predict(x_batch )             # shape (B, H, W, C)
        y_true.append(y_batch.numpy())                   # coleta targets
        y_pred.append(recon)

    y_true = np.concatenate(y_true, axis=0)               # (N, H, W, C)
    y_pred = np.concatenate(y_pred, axis=0)               # (N, H, W, C)



    # MSE por amostra: média do (y_true - y_pred)^2 em todas as pixels
    sample_losses.append({
        "split": split,
        "subset": subset,
        "loss": np.mean((y_true - y_pred) ** 2, axis=(1,2,3))
    })
        
titles = {
    # entropy subsets
    "split_1/subset_1": "0 to 2.28 entropy bits",
    "split_1/subset_2": "2.28 to 3.27 entropy bits",
    "split_1/subset_3": "3.27 to 4.21 entropy bits",
    "split_1/subset_4": "4.21 to inf entropy bits",
    "split_2/subset_1": "0 to 0.05 not black pixels (11 samples only)",
    "split_2/subset_2": "0.05 to 0.24 of not black pixels",
    "split_2/subset_3": "0.24 to 0.4 of not black pixels",
    "split_2/subset_4": "0.4 to 1 of not black pixels"
}

with plt.style.context(["science", "ieee"]):
    # Create a single figure with 8 boxplots in a row
    fig, axes = plt.subplots(1, len(sample_losses), figsize=(20, 5))  # 1 row, len(sample_losses) columns

    for i, sample_loss in enumerate(sample_losses):
        axes[i].boxplot(sample_loss["loss"], vert=True)
        axes[i].set_title(f"{titles[sample_loss['split'] + '/' + sample_loss['subset']]}")
        axes[i].set_ylabel("MSE")
        axes[i].set_ylim(0, 1.8)  # Fix y scale to [0, 1.8]
        axes[i].set_xticks([])  # No x-ticks needed for individual boxplots

    # Adjust layout and display the plots
    plt.tight_layout()
    fig.savefig("zz_ficheiro_bue_fixe.pdf", dpi=300)
    plt.show()
