# -----Imports-----
import numpy as np
import shutil

from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import load_img, img_to_array
from keras.models import load_model

# -----Create a base model-----
base_model = InceptionV3(
    include_top=False, 
    weights="imagenet",
    input_tensor=None,
    input_shape=(500, 500, 3), 
    pooling=None,
    classifier_activation="softmax"
)

# -----Freeze layers of the base model-----
for layer in base_model.layers:
    layer.trainable = False

# -----Downsamples the output of the the base model-----
X = GlobalAveragePooling2D()(base_model.output)

# -----Add hidden layers to learn new weights-----
X = Dense(units=400, activation="relu")(X)

# -----Add a output layer-----
X = Dense(units=3, activation="sigmoid")(X)

# -----Final model-----
model = Model(base_model.input, X)

# -----Compile the model-----
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# -----Setup batches of tensor image data with real-time data augmentation-----
data_gen = ImageDataGenerator(
    rescale=1./500.,
    featurewise_center=False,
    rotation_range=0.4,
    width_shift_range=0.3,
    horizontal_flip=True,
    preprocessing_function=preprocess_input,
    zoom_range=0.4,
    shear_range=0.4
)