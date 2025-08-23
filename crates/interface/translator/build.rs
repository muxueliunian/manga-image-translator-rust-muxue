fn main() {
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    #[cfg(target_os = "macos")]
    if std::env::var("BUILD_FOR_PYTHON").is_ok() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
    } else {
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
    }
}
