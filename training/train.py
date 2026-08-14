import os
import numpy as np
import keras
from keras import layers, Sequential
import tensorflow as tf
from tensorflow import data as tf_data
from pathlib import Path
import matplotlib.pyplot as plt
from keras.layers import Rescaling, RandomFlip, RandomRotation, RandomZoom
import pickle
import time


SEED = 42
IMAGE_SIZE = (128, 128)
INPUT_SIZE = (128, 128, 3)
BATCH_SIZE = 16

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data/processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_HISTORY_DIR = MODELS_DIR / "history"


train_ds = keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'train'),
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    pad_to_aspect_ratio=True,
    seed=SEED
)
validation_ds = keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'validation'),
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    pad_to_aspect_ratio=True,
    seed=SEED
)
# test_ds = keras.utils.image_dataset_from_directory(
#     os.path.join(DATA_DIR, 'test'),
#     batch_size=BATCH_SIZE,
#     image_size=IMAGE_SIZE,
#     pad_to_aspect_ratio=True,
#     seed=SEED
# )




preprocessing = Sequential([
    Rescaling(1./255),        # Scales pixel values from [0, 255] to [0, 1]
    # data_augmentation      # Your data augmentation block (already defined)
])
# Apply scaling and then data augmentation to the training images
# Apply preprocessing to the training dataset
train_ds = train_ds.map(
    lambda img, label: (preprocessing(img), label),
    num_parallel_calls=tf.data.AUTOTUNE# 'auto' works similarly to tf.data.AUTOTUNE in Keras
)

val_ds = validation_ds.map(
    lambda img, label: (img / 255.0, label),  # Manually scale without tf.cast
    num_parallel_calls=tf.data.AUTOTUNE
)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# test_ds = test_ds.map(
#     lambda img, label: (img / 255.0, label),  # Manually scale without tf.cast
#     num_parallel_calls=tf.data.AUTOTUNE
# )
# test_ds = test_ds.prefetch(tf.data.AUTOTUNE)


for images, labels in train_ds.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print(labels.numpy())




"""
ResNet18
MobileNetV3
EfficientNet-B0
ConvNeXt Tiny
ViT Tiny
"""
def add_final_layer(base_model, outputs_classes = 4, name="functional_model"):
    inputs = keras.Input(shape=INPUT_SIZE)
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(outputs_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs, name=name)

def create_model(imported_model, name):
    base_model = imported_model(weights='imagenet', include_top=False, input_shape=INPUT_SIZE)
    base_model.trainable = False

    final_model = add_final_layer(base_model, name=name)

    final_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    return final_model


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.total_start_time = time.time()
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.epoch_start_time
        self.epoch_times.append(duration)
        
        # INJECT INTO LOGS: Keras will automatically copy this into resnet_history.history
        if logs is not None:
            logs['epoch_time'] = duration

    def on_train_end(self, logs=None):
        self.total_time = time.time() - self.total_start_time
        print(f"\nTotal Training Time: {self.total_time:.2f} seconds")
        
        # OPTIONAL: Put total time into the final epoch logs if desired
        if logs is not None:
            logs['total_training_time'] = self.total_time


name = "ResNet"
print("="*50)
print(name)
print("="*50)

model_save_path = MODELS_HISTORY_DIR / f"{name}_history.pkl"
data_file_path = DATA_DIR / "dataset.csv"
time_callback = TimeHistory()


resnet_model = create_model(keras.applications.ResNet50V2, name)
resnet_model.summary()
resnet_history = resnet_model.fit(train_ds, epochs=5,  validation_data=val_ds, callbacks=[time_callback])

# Save the history dictionary to a file
with open(model_save_path, "wb") as file:
    pickle.dump(resnet_history, file)
