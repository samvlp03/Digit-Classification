# %%
#Importing the necessary modules
import numpy as np 
import tensorflow as tf 
from keras import layers, models

# Importing the MNIST dataset
from keras.datasets import mnist

from keras.utils import to_categorical
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt 
import matplotlib as mt 

# %%
# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# %%
# Building a convolutional neural network (CNN) 
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# %%
# Compiling the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# %%
# Training and evaluating the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)
predictions = model.predict(test_images)
test_loss, test_acc = model.evaluate(test_images, test_labels)

# %%
# Converting predictions into categorical data
predicted_labels = tf.argmax(predictions, axis=1)
true_labels = tf.argmax(test_labels, axis=1)

# %%
# Displaying classification report and accuracy
print('Classification Report:')
print(classification_report(true_labels, predicted_labels))
print(f'Test accuracy: {int(test_acc*100)}%')

# %%
num = int(input('Enter a number between 1 and 70000: '))
some_digit = train_images[num]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mt.cm.binary, interpolation='nearest')
plt.axis('off')
plt.show()


