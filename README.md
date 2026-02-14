# ROCm on Archlinux for RADEON RDNA Graphic Cards (pytorch, tensorflow)

#### For Docker version [read this](ROCm-archlinux-pytorch-tensorflow-docker.md)


## Standalone version:

#### Install `paru` to simplify installation and managing packages from the Arch User Repository (AUR)
```bash
sudo pacman -S --needed base-devel git
git clone https://aur.archlinux.org/paru.git
cd paru
makepkg -si
```
##### support packages
```bash
paru -S clang cmake
```

### ROCm Archlinux packages avaliable on AUR repositories
```bash
paru -S rocm-hip rocm-opencl-runtime rocm-hip-sdk rocm-opencl-sdk rocm-llvm
```
Configuring
```bash
sudo gpasswd -a username render
sudo gpasswd -a username video
```
### After ROCm installation check your gfx graphic card version
```
rocminfo | grep gfx
```
### Edit `.bashrc` to export ROCm and gfx variables
```
export ROCM_PATH=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=12.0.0
```
reopen your console

#### (Optional) If you want latest ROCm OpenCL (may cause some conflicts, but works):
Latest OpenCl (7.2.0) ROCM version - will conflict with rocm-opencl-runtime just accept both versions selecting 'No' option
```
paru -S rocm-opencl
```
select  the option `opencl-amd` or your specific gfx GPU if you want in this case `rocm-gfx120x-bin`
### 
### Check ROCm version installed
```
cat /opt/rocm/.info/version
```

## Test if ROCm are working in your Archlinux (download python script)
```bash
python test-rocm.py
```

### Support to HIP Ray Tracing and the open-source Graphics Library Framework
HIP: Heterogeneous-compute Interface for Portability - C++ API library to write code it run on AMD and NVIDA GPUs
```bash
paru -S hiprt glfw hip-runtime-amd
```

## ROCm and Pytorch

Install Pytorch for Archlinux ROCm
```bash
paru -S python-pytorch-rocm 
```
Install TensorFlow for Archlinux ROCm (many installation problems; consider using docker version instead)
```bash
paru -S bazelisk python-tensorflow-rocm
```

another packages
```bash
paru -S python-torchvision-rocm python-torchaudio-rocm
```

### Download Pytorch examples:
```bash
git clone git@github.com:pytorch/examples.git
```
Navigate to some pytorch example folder and create the python virtual environment (MNIST example in this case):
```bash
cd examples/mnist
```
Install torchvision package to run MNIST example:
```bash
sudo pacman -S python-torchvision
```

### Run pytorch `MNIST` example:

```
python main.py
```

### 1. Create a Virtual Environment:
```
python -m venv <my_venv_name>
```

### 2. Activate the Virtual Environment:
```
source <my_venv_name>/bin/activate
```

### 3. Install Dependencies from requirements.txt:
```
pip install -r requirements.txt
```

### 4. (Optional) Deactivate the Virtual Environment:
```
deactivate
```



## ROCm examples repository

These [instructions](https://github.com/ROCm/rocm-examples) assume that the prerequisites for every example are installed on the system.

### CMake

See CMake build options for an overview of build options.

```
git clone https://github.com/ROCm/rocm-examples.git
cd rocm-examples
cmake -S . -B build (on ROCm) or cmake -S . -B build -D GPU_RUNTIME=CUDA (on CUDA)
cmake --build build
cmake --install build --prefix install
```

### Make

Beware that only a subset of the examples support building via Make.

```
git clone https://github.com/ROCm/rocm-examples.git
cd rocm-examples
make (on ROCm) or $ make GPU_RUNTIME=CUDA (on CUDA)
```
    
Done for while!

## Reference links
https://wiki.archlinux.org/title/GPGPU
https://llm-tracker.info/howto/AMD-GPUs
https://gist.github.com/augustin-laurent/d29f026cdb53a4dff50a400c129d3ea7
[https://github.com/mpeschel10/test-tensorflow-rocm](https://github.com/mpeschel10/test-tensorflow-rocm)
https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html
