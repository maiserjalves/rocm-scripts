# Archlinux ROCm Podman Pytorch Tensorflow Jupyter notebook
```bash
sudo pacman -Syu
sudo pacman -S podman podman-desktop
```
### Podman Images to Pull
```
rocm/pytorch:latest
rocm/tensorflow:latest
```
### Run command
```bash
podman run -it --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --shm-size 4G \
    --security-opt seccomp=unconfined \
    rocm/pytorch:latest 
```

### Test tensorflow
```python
import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define a simple model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# Compile and train the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("Starting model fit...")
model.fit(x_train, y_train, epochs=1)
print("Model fit complete.")
```

