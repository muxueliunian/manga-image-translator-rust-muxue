#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Model creation error")]
    ModelCreate(#[from] ModelLoadError),
    #[error("before onnx model run")]
    Preprocess(#[from] PreProcessingError),
    #[error("onnx model issue")]
    Processing(#[from] ProcessingError),
    #[error("after onnx model run")]
    Postprocess(#[from] PostProcessingError),
}

#[derive(thiserror::Error, Debug)]
pub enum ModelLoadError {
    #[error("Failed to create model file")]
    CouldntCreateFile(#[from] std::io::Error),
    #[error("Failed to download model")]
    DownloadFailed(#[from] ureq::Error),
    #[cfg(feature = "onnx")]
    #[error("Failed to create session")]
    CreateSession(#[from] ort::Error),
    #[error("Model not registered")]
    ModelNotRegistered,
}

#[derive(thiserror::Error, Debug)]
pub enum ProcessingError {
    #[cfg(feature = "onnx")]
    #[error("caused by ort")]
    Model(#[from] ort::Error),
    #[error("caused by ort but shape mismatch")]
    Extract(#[from] ndarray::ShapeError),
}

#[derive(thiserror::Error, Debug)]
pub enum PreProcessingError {
    #[cfg(feature = "opencv")]
    #[error("caused by opencv")]
    OpenCv(#[from] opencv::Error),
    #[error("caused by ndarray")]
    NdArray(#[from] ndarray::ShapeError),
    #[error("expected input, but got none")]
    Empty,
}

#[derive(thiserror::Error, Debug)]
pub enum PostProcessingError {
    #[cfg(feature = "opencv")]
    #[error("caused by opencv")]
    OpenCv(#[from] opencv::Error),
    #[error("expected output, but got none")]
    Empty,
    #[error("caused by ndarray")]
    NdArray(#[from] ndarray::ShapeError),
}
