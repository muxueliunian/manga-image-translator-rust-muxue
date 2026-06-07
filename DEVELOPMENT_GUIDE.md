# Development Guide

本仓库当前以 Rust 版 `simple-runtime` 为主要开发目标，便携 GUI 使用 Windows WebView2 + `wry`，不是 Python WebUI。

## 主要入口

- WebView GUI：`crates/runtimes/simple-runtime/src/webview_ui.rs`
- WebView 静态资源：`crates/runtimes/simple-runtime/webview/`
- 原生 egui 备用界面：`crates/runtimes/simple-runtime/src/ui/`
- 执行流水线：`crates/runtimes/simple-runtime/src/execute/`
- 配置结构：`crates/runtimes/simple-runtime/src/settings/`
- PNG 嵌字渲染：`crates/modules/renderer/png/src/lib.rs`
- 便携打包：`scripts/package-windows-portable.ps1`

## WebView GUI 约定

- 前端通过 `window.ipc.postMessage` 向 `webview_ui.rs` 发送 IPC。
- IPC 类型定义在 `IpcKind`，新增按钮或异步行为时需要同步修改：
  - `webview/index.html`
  - `webview/app.js`
  - `src/webview_ui.rs`
- 翻译任务在后台线程运行，避免阻塞 UI。阶段进度通过 `UserEvent::Progress` 推送给前端。
- 翻译完成后结果先写入便携目录内部 `results/webview/job_*`，前端可预览并按复选框多选导出到用户选择的导出目录。
- API Key、OpenAI-compatible Base URL、模型名、prompt 等配置保存到便携目录 `config/app.json`。

## 渲染与排版

- GUI 的 `文字方向` 对应 `settings.render.text_direction`：
  - `Auto`
  - `Horizontal`
  - `Vertical`
- PNG 渲染器会根据原文特征和检测框比例做自动方向判断。遇到字幕被误判竖排时，优先调整 `auto_detect_vertical`。
- PNG 渲染器会对已放置文本做 AABB 碰撞避让，避免多个翻译块互相遮挡。若需要更精确，可升级为旋转矩形碰撞。

## Windows 构建环境

本地构建前通常需要设置 OpenCV 和 LLVM 路径：

```powershell
$env:OPENCV_LINK_LIBS='opencv_world4110'
$env:OPENCV_LINK_PATHS='C:\Users\atlas\Desktop\本子翻译\tools\opencv-4.11.0\opencv\build\x64\vc16\lib'
$env:OPENCV_INCLUDE_PATHS='C:\Users\atlas\Desktop\本子翻译\tools\opencv-4.11.0\opencv\build\include'
$env:OPENCV_DISABLE_PROBES='pkg_config,cmake,vcpkg_cmake,vcpkg'
$env:LIBCLANG_PATH='C:\Users\atlas\Desktop\本子翻译\tools\LLVM-22.1.6\bin'
$env:PATH='C:\Users\atlas\Desktop\本子翻译\tools\opencv-4.11.0\opencv\build\x64\vc16\bin;C:\Users\atlas\Desktop\本子翻译\tools\LLVM-22.1.6\bin;' + $env:PATH
```

常用命令：

```powershell
cargo fmt
cargo build -p simple-runtime --release --features cuda
.\scripts\package-windows-portable.ps1 -Cuda -NoZip
```

## 便携包验证

打包输出目录：

```text
dist/manga-image-translator-rust-portable/
```

推荐验证顺序：

1. 运行 `run-ui-debug.bat`，先看控制台日志。
2. 选择图片，确认列表中可以删除单个输入。
3. 选择 `PNG` 和 `文字方向`。
4. 开始翻译，确认进度条显示具体阶段。
5. 翻译完成后确认输入列表被清空。
6. 在结果区预览图片，勾选单张或多张，导出到指定目录。
7. 如 CUDA 未启用，检查日志中的 ONNX Runtime provider 和缺失 DLL。

## Git 约定

- 当前用户希望直接在 `master` 上开发并推送到 fork。
- fork remote 为 `muxue`，推送命令通常是：

```powershell
git push muxue master
```
