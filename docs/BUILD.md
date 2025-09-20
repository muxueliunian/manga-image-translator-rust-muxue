# Build

### Preperation All
Install rust with [rustup](https://rustup.rs)
Install [cuda](https://developer.nvidia.com/cuda-12-9-0-download-archive)
Install [cudnn](https://developer.nvidia.com/cudnn-downloads)

### Preparation Ubuntu/Debian
```sh
sudo apt-get update
sudo apt-get install -y pkg-config libssl-dev libopencv-dev clang libclang-dev libfontconfig-dev
```

### Preperation MacOS
```sh
brew install llvm opencv
# Old macs only
brew install openssl@3
# Run this on every terminal session(not actually required for debug builds/only release builds)
export OPENCV_LINK_LIBS=opencv_core,opencv_imgproc,opencv_calib3d
```

### Preperation Windows
```sh
choco install opencv llvm

$env:OPENCV_LINK_LIBS = $libName # opencv_world*.lib. Its the only .lib file in the C:\tools\opencv if you use the prebuilts
$env:OPENCV_LINK_PATHS = $libPath # the parent folder of the opencv_world*.lib file. maybe "C:\tools\opencv\build\x64\vc16\lib"
$env:OPENCV_INCLUDE_PATHS = $includePath # most likely "C:\tools\opencv\build\include"
$env:Path = "C:\tools\opencv\build\x64\vc16\bin;" + $env:Path

or permanent

[System.Environment]::SetEnvironmentVariable("OPENCV_LINK_LIBS", "opencv_world4110", "User")
[System.Environment]::SetEnvironmentVariable("OPENCV_LINK_PATHS", "C:\tools\opencv\build\x64\vc16\lib", "User")
[System.Environment]::SetEnvironmentVariable("OPENCV_INCLUDE_PATHS", "C:\tools\opencv\build\include", "User")
[System.Environment]::SetEnvironmentVariable("Path", "C:\tools\opencv\build\x64\vc16\bin;" + $env:Path, "User")
```

[Path to long error 1](https://stackoverflow.com/questions/22575662/filename-too-long-in-git-for-windows)
[Path to long error 2](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry)

## Quick Start
```sh
git clone https://github.com/frederik-uni/manga-image-translator-rust --recursive

cargo r -p simple-runtime -- -i in -o out
```


## Dependencies
- [rustup](https://rustup.rs)
- openssl/libssl-dev
- [opencv](https://github.com/twistedfall/opencv-rust/blob/master/INSTALL.md)
- libfontconfig-dev(linux only)
- clang libclang-dev(linux only)
- llvm(macos/windows only)


# Deploy
- opencv
- onnxruntime exectuion providers
- main binary

# CPP Dependencies

- [opencv](https://docs.opencv.org/4.x/index.html)
- [ort/onnxruntime](https://github.com/microsoft/onnxruntime)
- [clipper2](https://github.com/AngusJohnson/Clipper2)
