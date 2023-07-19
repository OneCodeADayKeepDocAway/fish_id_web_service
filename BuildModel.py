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