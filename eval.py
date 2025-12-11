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
import scienceplots


plt.style.use('science')

keras.backend.clear_session() # Clear any previous session

DATASET_BUFFER_SIZE = 1000
DATASET_BATCH_SIZE = 32

IMG_WIDTH = 256
IMG_HEIGHT = 256

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

train_dataset = make_ds('./data/TCIRRP/train1k/*.jpg', training=True)
test_dataset  = make_ds('./data/TCIRRP/test/*.jpg',  training=False)


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


## Model instantiation
input_model_path = f"60k-55Epochs-64Batch-32Lat.keras"

print("\033[92mModel loaded. With evaluation:\033[0m")
autoencoder = load_model(input_model_path, custom_objects={'Autoencoder': Autoencoder})
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
sample_losses = np.mean((y_true - y_pred) ** 2, axis=(1,2,3))  # vetor de tamanho N

# 1) Calcular IQR e threshold de outlier
q1, q3 = np.percentile(sample_losses, [25, 75])
iqr = q3 - q1
threshold = q3 + 1.5 * iqr

# 2) Indices com a perda maior (os 5 maiores)
outlier_idxs = np.argsort(sample_losses)[-5:]

# 2.1) Indices com a perda menor (os 5 menores)
lower_idxs = np.argsort(sample_losses)[:5]

# # 3) Mostrar alguns desses exemplos
# if len(outlier_idxs) != 0:
#     num_show = min(5, len(outlier_idxs))
#     fig, axes = plt.subplots(num_show, 2, figsize=(6, 2*num_show))
#     for i, idx in enumerate(outlier_idxs[:num_show]):
#         axes[i,0].imshow((y_true[idx].squeeze()), cmap='gray')
#         axes[i,0].set_title(f"Target #{idx}\nMSE={sample_losses[idx]:.2f}")
#         axes[i,0].axis('off')
        
#         axes[i,1].imshow((y_pred[idx].squeeze()), cmap='gray')
#         axes[i,1].set_title(f"Predicted #{idx}")
#         axes[i,1].axis('off')

#     plt.tight_layout()
#     plt.show()
    
# # 3.1) Mostrar alguns exemplos de imagens com perda baixa
# if len(lower_idxs) != 0:
#     num_show = min(5, len(lower_idxs))
#     fig, axes = plt.subplots(num_show, 2, figsize=(6, 2*num_show))
#     for i, idx in enumerate(lower_idxs[:num_show]):
#         axes[i,0].imshow((y_true[idx].squeeze()), cmap='gray')
#         axes[i,0].set_title(f"Target #{idx}\nMSE={sample_losses[idx]:.2f}")
#         axes[i,0].axis('off')
        
#         axes[i,1].imshow((y_pred[idx].squeeze()), cmap='gray')
#         axes[i,1].set_title(f"Predicted #{idx}")
#         axes[i,1].axis('off')

#     plt.tight_layout()
#     plt.show()




# with plt.style.context(["science", "ieee"]):
#     # --- Create a single figure with two subplots for this experiment ---
#     fig, axes = plt.subplots(1, 2, figsize=(7, 3)) # 1 row, 2 columns

#     # Left subplot: Boxplot
#     axes[0].boxplot(mse_scores, vert=True, widths=0.5)
#     axes[0].set_title("Boxplot")
#     axes[0].set_ylabel("MSE")
#     axes[0].set_xticks([]) # No x-ticks needed for a single boxplot

#     # Right subplot: Histogram
#     axes[1].hist(mse_scores, bins=30, edgecolor='black', color='skyblue', alpha=0.8)
#     # For exact Keras default histogram facecolor, remove color and alpha:
#     # axes[1].hist(mse_scores, bins=30, edgecolor='black')
#     axes[1].set_title("Histogram")
#     axes[1].set_xlabel("MSE")
#     axes[1].set_ylabel("Number of Images") # Or "Frequency"
#     axes[1].ticklabel_format(axis='x', style='sci', scilimits=(-2,2), useMathText=True) # Scientific notation if needed

#     # fig.suptitle(f"MSE Analysis: {exp_id}", y=1.0)

#     plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout

#     plot_filename_pdf = os.path.join(output_dir, f"mse_boxplot_hist_{exp_id}.pdf")
#     plot_filename_png = os.path.join(output_dir, f"mse_boxplot_hist_{exp_id}.png")
#     plt.savefig(plot_filename_pdf, dpi=300)
#     plt.savefig(plot_filename_png, dpi=300)
#     plt.close(fig)

with plt.style.context(["science", "ieee"]):
    # Create a single figure with two subplots
    fig, axes = plt.subplots(1, 2)  # 1 row, 2 columns

    # Subplot 1: Boxplot of losses
    axes[0].boxplot(sample_losses, vert=True)
    axes[0].set_title("Boxplot")
    axes[0].set_ylabel("MSE")
    axes[0].set_xticks([])  # No x-ticks needed for a single boxplot

    # Subplot 2: Histogram of losses
    axes[1].hist(sample_losses, bins=25, edgecolor='black', color='skyblue', alpha=0.8)
    axes[1].set_title("Histogram")
    axes[1].set_xlabel("MSE")
    axes[1].set_ylabel("Number of Images")

    # Adjust layout and display the plots
    plt.tight_layout()
    fig.savefig("zz_ficheiro_bue_fixe.pdf", dpi=300)
    plt.show()
