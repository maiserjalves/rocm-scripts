# Archlinux ROCm Docker Pytorch Tensorflow Jupyter notebook
```bash
sudo pacman -Syu
```
### Docker
```bash
sudo pacman -S docker docker-compose
```

add docker in system service
```bash
sudo systemctl start docker.service
sudo systemctl enable docker.service
```
run docker witout sudo
```bash
sudo usermod -aG docker $USER
```
you need to log out and log back in, or restart your system, for the group membership to be re-evaluated
```bash
docker run hello-world
```
### Pull ROCm pytorch and tensorflow docker images
```bash
docker pull rocm/pytorch:latest
docker pull rocm/tensorflow:latest
```

### Run docker command to instantiate `pytorch` rocm container
```bash
docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 4G -v $HOME/dockerx:/dockerx -w /dockerx rocm/pytorch:latest
```

### Run docker command to instantiate `tensorflow` rocm container
```bash
docker run -it --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 4G --group-add=video --cap-add=SYS\_PTRACE --security-opt seccomp=unconfined -v $HOME/dockerx:/dockerx -w /dockerx rocm/tensorflow:latest /bin/bash
```


#### params:
* `--network=host`: Allows you to access the container's services (like Jupyter) on the same port on your host machine.

* `--device=/dev/kfd --device=/dev/dri --group-add=video`: These flags are crucial for providing the container access to the host's AMD GPU hardware.

* `-v $HOME/dockerx:/dockerx`: Mounts a local directory for persistence.

* `-w /dockerx`: Sets the working directory inside the container.



### Test working rocm in containers


for `rocm/pytorch` container
```bash
python -c "import torch; print(f'torch version: {torch.__version__}'); print(f'ROCm available: {torch.cuda.is_available()}')"
```


for `rocm/tensorflow` container
```bash
python -c 'import tensorflow as tf; print("Built with ROCm support:", tf.test.is_built_with_rocm()); print("Available GPUs:", tf.config.list_physical_devices("GPU"))'
```



### Install Jupyter Notebook inside the pytorch or tensorflow rocm container
```
pip install jupyter notebook
```
### Launch Jupyter Notebook Inside the Container 
```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```
open Jupyter in Browser 127.0.0.1...


#### snapshoting docker container after modifications
```bash
docker ps -a
docker commit <containerID> <new_image_name>:<tag>

example: docker commit <containerID> rocm/pytorch:latest-with-jupyter
example: docker commit <containerID> rocm/tensorflow:latest-with-jupyter
```
## Test scripts

#### for Pytorch test script
```python
import torch
print("PyTorch Version:", torch.__version__)
print("ROCm Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0))
```

#### for TensorFlow test script
```python
import tensorflow as tf;
print("TensorFlow Version:", tf.__version__)
print("Built with ROCm support:", tf.test.is_built_with_rocm())
print("Available GPUs:", tf.config.experimental.get_device_details(tf.config.list_physical_devices("GPU")[0])['device_name'])
```

#### pytorch MNIST dataset test
```python
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


class Args(argparse.Namespace):
    batch_size = 64
    test_batch_size = 1000
    epochs = 14
    lr = 1.0
    gamma = 0.7
    no_accel = False
    dry_run = False
    seed = 1
    log_interval = 10
    save_model = True


def main():
    args=Args()
    use_accel = not args.no_accel and torch.accelerator.is_available()
    torch.manual_seed(args.seed)

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")
    print(device)

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    if use_accel:
        accel_kwargs = {'num_workers': 1,
                        'persistent_workers': True,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == '__main__':
    main()
```

#### tensorflow MNIST dataset test
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
