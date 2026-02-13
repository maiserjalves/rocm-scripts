#!/bin/python3

# You can use environment variables if your GPU works, but is not officially supported.
# For example, I have a Radeon RX 6750 XT, which is detected as the unsupported gfx1031 arch,
#  but the supported gfx1030 architecture works just dandy.
#import os
#os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

import tensorflow as tf
# If you're really having trouble, turn on set_log_device_placement.
# This will print a bunch of gobbledegook,
#  but at the end of every line, it will say what device something is located on.
#tf.debugging.set_log_device_placement(True)

# tf.device('/GPU:0') fails silently if the device does not exist,
#  and otherwise does not appear to honor requests e.g. to use CPU instead of GPU.
# Do not trust it.
# You can use os.environ['HSA_OVERRIDE_GFX_VERSION'] = 'dummy' to force CPU usage for comparison.
if not len(tf.config.list_physical_devices('GPU')) > 0:
    raise Exception('tf.config.list_physical_devices(\'GPU\') returns empty list; no GPUs found')
print(tf.config.list_physical_devices())

import time
start = time.time()

# Taken from tensorflow.org/tutorials/quickstart/beginner.
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255., x_test / 255.

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # This should run REALLY slow on CPU, but fast enough on GPU.
    tf.keras.layers.Dense(1280, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10),
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
prediction = model(x_train[:1])
print('Expected loss: 2.3. Actual: ', loss_fn(y_train[:1], prediction))
print('Probabilities: ', tf.nn.softmax(prediction))

model.compile(loss=loss_fn, optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)

end = time.time()
print('Your run took %f seconds. ' % (end - start))
print('My GPU takes 14 seconds.')
print('My CPU takes 74 seconds.')
print('Your mileage may vary.')

