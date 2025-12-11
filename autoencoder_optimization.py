import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
from tensorflow.keras.models import Model, load_model
from keras.datasets import mnist
from itertools import product
from tensorflow.data import AUTOTUNE
import os
import math
import random
import scienceplots

plt.style.use('science')
# keras.backend.clear_session() # Clear any previous session

batch_sizes = (4, 8, 16, 32, 64)
last_activations = ("linear", "relu", "sigmoid", "tanh")

parameter_combinations = list(product(batch_sizes, last_activations))

# results = []
results = [(4, 'linear', 0.6332380771636963), (4, 'relu', 0.7101576328277588), (4, 'sigmoid', 0.7978470921516418), (4, 'tanh', 0.7093089818954468), (8, 'linear', 0.6341965198516846), (8, 'relu', 0.6934881806373596), (8, 'sigmoid', 0.7924292087554932), (8, 'tanh', 0.7091810703277588), (16, 'linear', 0.6408583521842957), (16, 'relu', 0.7002402544021606), (16, 'sigmoid', 0.7762771248817444), (16, 'tanh', 0.7078699469566345), (32, 'linear', 0.6457149386405945), (32, 'relu', 0.7086055278778076), (32, 'sigmoid', 0.7822685241699219), (32, 'tanh', 0.7048413157463074), (64, 'linear', 0.6388959288597107), (64, 'relu', 0.7021109461784363), (64, 'sigmoid', 0.8442230820655823), (64, 'tanh', 0.7114475965499878)]

# num_epochs = 20

# for chosen_parameters in parameter_combinations:
#     print(f"Running test with parameters: {chosen_parameters}")
    
#     batch_size = int(chosen_parameters[0])
#     last_activation = chosen_parameters[1]
    
#     DATASET_BUFFER_SIZE = 1000
#     DATASET_BATCH_SIZE = batch_size

#     IMG_WIDTH = 256
#     IMG_HEIGHT = 256

#     # 1) Camadas de augmentation específicas
#     data_augment = tf.keras.Sequential([
#         # rotações entre 0 e 360 graus
#         layers.RandomRotation(factor=1.0, fill_mode='reflect'),
#         # zoom in/out leve (±10%)
#         layers.RandomZoom(0.1, 0.1, fill_mode='reflect'),
#         # contrate variações leves
#         layers.RandomContrast(0.05),
#         # brilho leve
#         layers.RandomBrightness(0.05),
#     ])

#     def preprocess_pair(path, training):
#         # a) carregar e split
#         img = tf.io.decode_jpeg(tf.io.read_file(path), channels=1)  # já grayscale
#         w = tf.shape(img)[1] // 2
#         # “real” é a metade esquerda, “inp” a direita
#         real = img[:, :w, :]
#         inp  = img[:, w:, :]

#         # b) augment só em treino
#         if training:
#             # concatenamos para aplicar transform igual em ambos
#             pair = tf.concat([inp, real], axis=2)  # shape (H, W, 2)
#             pair = data_augment(pair)
#             inp, real = tf.split(pair, num_or_size_splits=2, axis=2)

#         # c) resize com bilinear + padding reflexivo p/ evitar artefatos
#         inp  = tf.image.resize(inp,  [IMG_WIDTH, IMG_HEIGHT], method='bilinear')
#         real = tf.image.resize(real, [IMG_WIDTH, IMG_HEIGHT], method='bilinear')

#         # d) z-score normalization por imagem
#         inp_mean, inp_var = tf.nn.moments(inp, axes=[0,1])
#         real_mean, real_var = tf.nn.moments(real, axes=[0,1])
#         inp  = (inp  - inp_mean)  / tf.sqrt(inp_var  + 1e-6)
#         real = (real - real_mean) / tf.sqrt(real_var + 1e-6)

#         return inp, real

#     def make_ds(pattern, training):
#         ds = tf.data.Dataset.list_files(pattern)
#         if training:
#             ds = ds.shuffle(DATASET_BUFFER_SIZE)
#         ds = ds.map(lambda p: preprocess_pair(p, training),
#                     num_parallel_calls=AUTOTUNE)
#         ds = ds.batch(DATASET_BATCH_SIZE, drop_remainder=True)
#         ds = ds.prefetch(AUTOTUNE)
#         return ds

#     train_dataset = make_ds('./data/TCIRRP/train1k/*.jpg', training=True)
#     test_dataset  = make_ds('./data/TCIRRP/test0.1k/*.jpg',  training=False)
#     test_dataset_full = make_ds('./data/TCIRRP/test/*.jpg', training=False)


#     ######################################################################

#     LATENT_DIMENSIONS = 32
        
#     class Autoencoder(Model):
#         def __init__(self, **kwargs):
#             super(Autoencoder, self).__init__(**kwargs)
            
#             # Apply convulutional layers and move from flatten to latent hidden layer
#             self.encoder = tf.keras.Sequential([
#                 layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1)), # one channel (intensity)
                
#                 # 1ª camada conv
#                 layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False),
#                 layers.BatchNormalization(),
#                 layers.Activation('relu'),
                
#                 # 2ª camada conv
#                 layers.Conv2D(64, 3, strides=2, padding='same', use_bias=False),
#                 layers.BatchNormalization(),
#                 layers.Activation('relu'),
                
#                 # 3ª camada conv
#                 layers.Conv2D(128, 3, strides=2, padding='same', use_bias=False),
#                 layers.BatchNormalization(),
#                 layers.Activation('relu'),
        
#                 layers.Flatten(),
#                 layers.Dense(LATENT_DIMENSIONS, activation='relu'),
#             ])
            
#             self.bw = math.ceil(IMG_WIDTH / 2**3)   # width bottleneck
#             self.bh = math.ceil(IMG_HEIGHT / 2**3)  # height bottleneck
#             self.bc = 128                           # channels bottleneck
                        
#             self.decoder = tf.keras.Sequential([
#                 layers.Input(shape=(LATENT_DIMENSIONS,)),
                
#                 layers.Dense(self.bw * self.bh * self.bc, use_bias=False),
#                 layers.BatchNormalization(),
#                 layers.Activation('relu'),
                
#                 layers.Reshape((self.bw, self.bh, self.bc)),
                
#                 layers.Conv2DTranspose(64, 3, strides=2, padding='same', use_bias=False),
#                 layers.BatchNormalization(),
#                 layers.Activation('relu'),
                
#                 layers.Conv2DTranspose(32, 3, strides=2, padding='same', use_bias=False),
#                 layers.BatchNormalization(),
#                 layers.Activation('relu'),
                
#                 layers.Conv2DTranspose(1, 3, strides=2, padding='same', activation=last_activation),
#             ])
        
#         def call(self, input_data):
#             encoded = self.encoder(input_data)
#             decoded = self.decoder(encoded)
#             return decoded


#     ## Model instantiation

#     input_epoch = 0
#     input_model_path = f"0.01k-{input_epoch}Epochs-32Lat-{batch_size}Batch-{last_activation}.keras"
#     output_epoch = num_epochs
#     output_model_path = f"0.01k-{output_epoch}Epochs-32Lat-{batch_size}Batch-{last_activation}.keras"
#     log_dir = f"logs/fit/v2-autoencoder-0.01k-32Lat-{last_activation}-{batch_size}Batch"

#     if os.path.exists(input_model_path):
#         print("\033[92mModel loaded. With evaluation:\033[0m")
#         autoencoder = load_model(input_model_path, custom_objects={'Autoencoder': Autoencoder})
#         autoencoder.evaluate(test_dataset, verbose=2)
#         print(autoencoder.optimizer.get_config())

#     else:
#         print("Model not loaded")
#         autoencoder = Autoencoder()
#         autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=[tf.keras.metrics.Accuracy()])


#     # autoencoder.encoder.summary()
#     # autoencoder.decoder.summary()

#     ## Model training

#     save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=output_model_path, verbose=2, monitor='val_loss')

#     tensorboard_cb = TensorBoard(
#         log_dir=log_dir,
#         histogram_freq=5,      # grava histogramas de pesos a cada época
#         write_graph=True,      # salva a definição do grafo
#     )

#     autoencoder.fit(
#         train_dataset,
#         epochs=output_epoch,
#         initial_epoch=input_epoch,
#         validation_data=test_dataset,
#         callbacks=[save_callback, tensorboard_cb],
#     )

#     ## Model Testing

#     print("\n\n\n")

#     print("Final evaluation:")
#     ev = autoencoder.evaluate(test_dataset_full, verbose=2)
#     results.append((batch_size, last_activation, ev[0]))  # ev[0] is the loss value

#     print("Final evaluation result:", ev)
    

print(results)

# set up the figure and Axes
fig = plt.figure()

with plt.style.context(["science", "ieee"]):

    _x = np.array(batch_sizes)
    _y = np.array(last_activations)

    heatmap_data = np.zeros((len(_y), len(_x)))
    for idx, (batch_sz, last_act, val_loss) in enumerate(results):
        i = batch_sizes.index(batch_sz)
        j = last_activations.index(last_act)
        heatmap_data[j, i] = val_loss  # Assign the validation loss to the heatmap

    # Create heatmap
    plt.imshow(heatmap_data, cmap='gray', origin='lower', aspect='auto')
    plt.colorbar(label='Mean Squared Error (MSE)')
    # plt.title('Avg MSE: Batch Size vs. Last Activation Function')
    plt.xlabel('Batch Size')
    plt.ylabel('Last Activation Function')
    plt.xticks(ticks=np.arange(len(batch_sizes)), labels=batch_sizes)
    plt.yticks(ticks=np.arange(len(last_activations)), labels=last_activations)

    # Add values above the cells
    for i in range(len(_y)):
        for j in range(len(_x)):
            if (i == 2 and j == 4):
                plt.text(j, i, f"{heatmap_data[i, j]:.2f}", ha='center', va='center', color='black')
                continue

            plt.text(j, i, f"{heatmap_data[i, j]:.2f}", ha='center', va='center', color='white')

    fig.savefig("teste.pdf", dpi=300)

    plt.show()


# num_tests = 5

# for name, dataset in [("Training set", train_dataset), ("Testing set", test_dataset)]:
#     # 1) Extrair algumas amostras
#     x_input = []
#     x_real  = []
#     for input_batch, real_batch in dataset.take(num_tests):
#         # Assumindo batch_size = 1
#         x_input.extend(input_batch)
#         x_real.extend(real_batch)
#     # empilha em tensores [num_tests * batch_size, ...]
#     x_input = tf.stack(x_input)
#     x_real  = tf.stack(x_real)

#     # 2) Codifica e decodifica
#     encoded_imgs = autoencoder.encoder(x_input).numpy()
#     decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

#     # 3) Plota
#     plt.figure(figsize=(6, num_tests * 2))
#     plt.suptitle(name, fontsize=16)
#     for i in range(num_tests):
#         # Coluna 1: original de entrada
#         ax = plt.subplot(num_tests, 3, i*3 + 1)
#         plt.imshow(x_input[i].numpy(), cmap="gray")
#         ax.set_title("Reversed")
#         ax.axis("off")

#         # Coluna 2: reconstruída
#         ax = plt.subplot(num_tests, 3, i*3 + 2)
#         plt.imshow(decoded_imgs[i], cmap="gray")
#         ax.set_title("Preddiction")
#         ax.axis("off")

#         # Coluna 3: imagem “real” alvo (se for diferente da entrada)
#         ax = plt.subplot(num_tests, 3, i*3 + 3)
#         plt.imshow(x_real[i].numpy(), cmap="gray")
#         ax.set_title("Target")
#         ax.axis("off")
                

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()
