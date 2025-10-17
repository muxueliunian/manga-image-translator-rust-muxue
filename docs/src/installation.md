# Installation
Binaries are available [here](https://github.com/frederik-uni/manga-image-translator-rust/releases/latest/) for windows, linux and MacOs for arm64 and x86_64

For faster execution, it is recommended to install CUDA and cuDNN.

Install [cuda 12.9](https://developer.nvidia.com/cuda-12-9-0-download-archive)

Install [cudnn](https://developer.nvidia.com/cudnn-downloads)


If you use cuda delete the `cudnn*` file in the downloaded folder.
Otherwise, delete the `onnxruntime cuda execution provider`

## Linux only
- `cd path/to/folder`
- `echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)" >> ~/.bashrc`
- `echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)" >> ~/.zshrc`
