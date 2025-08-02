# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


        # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# import dataframes
train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
#print(train_df.head())
#print(test_df.head())


X = train_df.drop('label', axis=1)
y = train_df['label']
# Set up train/validate split
X_train, X_valid, y_train, y_valid =  train_test_split(X, y, random_state=0)
#print(X_train.head())
#print(y_train.head())


# Visualize the first 5 images
plt.figure(figsize=(10, 2))
for i in range(5):
    image = X.iloc[i].values.reshape(28, 28)  # reshape to 28x28
    label = y.iloc[i]
    plt.subplot(1, 5, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Data Pipeline

# Normalise pixel values between [0, 1]
X_train = X_train / 255.0
X_valid = X_valid / 255.0

# Convert from pandas dataframe to tensor flow dataset
train_ds = tf.data.Dataset.from_tensor_slices((X_train.values.reshape(-1, 28, 28, 1), y_train.values))
valid_ds = tf.data.Dataset.from_tensor_slices((X_valid.values.reshape(-1, 28, 28, 1), y_valid.values))

# Apply batching, shuffling, caching
BATCH_SIZE = 64
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = (
    train_ds
    .shuffle(buffer_size=1024)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

valid_ds = (
    valid_ds
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

# apply augmentation to improve dataset variability
# dont use vertical or horizontal flips since that could cause 6's and 9's to be confused, 
# also don't allow more than a 10% rotation for the same reason
augment = keras.Sequential([
    layers.RandomContrast(factor=0.5),
    layers.RandomWidth(factor=0.15), # horizontal stretch
    layers.RandomHeight(factor=0.15),
    layers.RandomRotation(factor=0.10),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.Resizing(28, 28)
])

# apply augmentation to training dataset only:
train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y))

ex = next(iter(train_ds.unbatch().map(lambda x, y: x).batch(1)))


# visualise the augmented dataset
plt.figure(figsize=(10,10))
for i in range(16):
    image = augment(ex, training=True)
    plt.subplot(4, 4, i+1)
    plt.imshow(tf.squeeze(image))
    plt.axis('off')
plt.show()

# Build the model
model = keras.Sequential([
    layers.InputLayer(input_shape=[28, 28, 1]),

    layers.BatchNormalization(),
    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),

    layers.BatchNormalization(),
    layers.Conv2D(128, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),

    layers.BatchNormalization(),
    layers.Conv2D(256, 3, activation='relu', padding='same'),
    layers.Conv2D(256, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),

    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax'),
])


# Train the model
optimizer = tf.keras.optimizers.Adam(epsilon=0.01)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'],
)

early_stop_cb = keras.callbacks.EarlyStopping(
    patience=5,
    restore_best_weights=True,
    monitor="val_sparse_categorical_accuracy",
    mode="max"
)

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=50,
    callbacks=[early_stop_cb]
)

# Plot learning curves
import pandas as pd
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['sparse_categorical_accuracy', 'val_sparse_categorical_accuracy']].plot();


# set up the full dataset to train the model on the most data
# for the test output:
X_full = train_df.drop('label', axis=1) / 255.0
X_full_reshaped = X_full.values.reshape(-1, 28, 28, 1)

y_full = train_df['label']

full_ds = tf.data.Dataset.from_tensor_slices((X_full_reshaped, y_full.values))
full_ds = (
    full_ds
    .shuffle(buffer_size=1024)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

# retrain using full dataset
optimizer = tf.keras.optimizers.Adam(epsilon=0.01)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'],
)

early_stop_cb = keras.callbacks.EarlyStopping(
    patience=5,
    restore_best_weights=True,
    monitor="val_sparse_categorical_accuracy",
    mode="max"
)

history = model.fit(
    full_ds,
    epochs=50,
    callbacks=[early_stop_cb] 
)

# generating submission from test data
X_test = test_df / 255.0
X_test_reshaped = X_test.values.reshape(-1, 28, 28, 1)

predictions = model.predict(X_test_reshaped)
predicted_labels = np.argmax(predictions, axis=1)

submission = pd.DataFrame({
    'ImageId': np.arange(1, len(predicted_labels) + 1),
    'Label': predicted_labels
})
submission.to_csv('submission.csv', index=False)


# Show 5 examples from the test set with predicted labels
print('Examples of test images with model applied labels:')
plt.figure(figsize=(10, 2))
for i in range(5):
    image = X_test_reshaped[i].reshape(28, 28)
    label = predicted_labels[i]
    plt.subplot(1, 5, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"Pred: {label}")
    plt.axis('off')
plt.tight_layout()
plt.show()