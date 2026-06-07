# Manga Image Translator

*   [Docs/Install Guide](https://frederik-uni.github.io/manga-image-translator-rust/index.html)
*   [Config file example](example/example.json)
*   [Config file schema](example/schema.json)

## Windows portable package

Build a portable Windows folder from PowerShell:

```powershell
.\scripts\package-windows-portable.ps1
```

The script creates `dist\manga-image-translator-rust-portable\` and, by default,
`dist\manga-image-translator-rust-portable.zip`. Use `-NoZip` to skip the zip,
or `-Cuda` to build with the CUDA feature:

```powershell
.\scripts\package-windows-portable.ps1 -Cuda
.\scripts\package-windows-portable.ps1 -NoZip
```

The portable folder includes `run-ui.bat`, `run-webui.bat`,
`run-cli-example.bat`, `README-portable.txt`, and the `config`, `models`,
`uploads`, and `results` directories.
