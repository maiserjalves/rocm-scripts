# ROCm on Archlinux for RADEON RDNA Graphic Cards
### Install `paru` to simplify installation and managing packages from the Arch User Repository (AUR)
```
sudo pacman -S --needed base-devel git
git clone https://aur.archlinux.org/paru.git
cd paru
makepkg -si
```

### ROCm Archlinux packages avaliable on AUR repositories
```
paru -S rocm-hip rocm-opencl-runtime rocm-hip-sdk rocm-opencl-sdk rocm-llvm
```

### (Optional) If you want latest ROCm OpenCL (may cause some conflicts, but works):
Latest OpenCl (7.0.2) ROCM version - will conflict with rocm-opencl-runtime just accept both versions selecting 'No' option
```
paru -S rocm-opencl
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

## Test if ROCm are working in your Archlinux (download python script)
```
wget https://gist.githubusercontent.com/damico/484f7b0a148a0c5f707054cf9c0a0533/raw/43c317bfbde626d9112d44462d815613194988e0/test-rocm.py
```

```
python test-rocm.py
```

### Support to HIP Ray Tracing and the open-source Graphics Library Framework
```
paru -S hiprt glfw hip-runtime-amd
```

## ROCm and Pytorch

Install Pytorch for Archlinux ROCm
```
paru -S python-pytorch-rocm
```

### Download Pytorch examples:
```
git clone git@github.com:pytorch/examples.git
```

Navigate to some pytorch example folder and create the python virtual environment (MNIST example in this case):

```
cd examples/mnist
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

### Run pytorch `MNIST` example:
```
python main.py
```

## Rocm examples repository

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

https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html
