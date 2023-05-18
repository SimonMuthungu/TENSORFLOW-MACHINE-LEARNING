import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import urllib.request
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip' 

file_name = 'horse-or-human.zip'
training_dir = 'horse-or-human/training/'
urllib.request.urlretrieve(url, file_name)

zip_ref = zipfile.Zipfile(file_name, 'r')
zip_ref.extract_all(training_dir)
zipref.close()

train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size = (300, 3000),
    class_mode = 'binary'
)

# defining our model
model = Sequential([
    tf.keras.layers.conv2D(16, (3, 3), input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.conv2D(32, (3, 3), input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.conv2D(64, (3, 3), input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.conv2D(64, (3, 3), input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.conv2D(64, (3, 3), input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
                    ])
model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(
    train_generator,
    epochs=15
)
