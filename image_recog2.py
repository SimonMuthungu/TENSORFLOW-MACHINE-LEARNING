import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# were getting alerted when accuracy from the metrics hits 95
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if logs.get('accuracy')>95:
      print("\nWere very accurate, so were sotpping")
      self.model.stop_training = True 

callbacks = myCallback()
data = tf.keras.datasets.fashion_mnist
(training_data, training_labels), (test_data, test_labels) = data.load_data()

training_data = training_data / 255.0
test_data = test_data / 255.0

# defining our model
model = Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
                    Dense(128, activation=tf.nn.relu),
                    Dense(10, activation=tf.nn.softmax)
                    ])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_data, training_labels, epochs=50, callbacks=[callbacks])
# model.evaluate(test_data, test_labels)
