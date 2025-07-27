# Build

## Dependencies

- openssl
- [opencv](https://github.com/twistedfall/opencv-rust/blob/master/INSTALL.md)
- libfontconfig-dev(linux only)

# Links

## Detectors

| Model          | Paper                                                                               | Train                                                                                                  | Source                                                                                |
| -------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| dbnet          | [ARXIV](https://arxiv.org/abs/1911.08947) [ARXIV](https://arxiv.org/abs/2202.10304) | /                                                                                                      | [GitHub](https://github.com/zyddnys/manga-image-translator/tree/main/models/dbnet)    |
| ctd            | /                                                                                   | /                                                                                                      | [GitHub](https://github.com/zyddnys/manga-image-translator/tree/main/models/ctd)      |
| dbnet_convnext | /                                                                                   | /                                                                                                      | [GitHub](https://github.com/zyddnys/manga-image-translator/tree/main/models/convnext) |
| Paddle         | /                                                                                   | [Docs](https://paddlepaddle.github.io/PaddleOCR/main/en/version2.x/ppocr/model_train/recognition.html) | [GitHub](https://github.com/PaddlePaddle/PaddleOCR)                                   |

## OCRs

| Model     | Paper | Train                                                                    | Source                                           |
| --------- | ----- | ------------------------------------------------------------------------ | ------------------------------------------------ |
| manga-ocr | /     | [Docs](https://github.com/kha-white/manga-ocr/tree/master/manga_ocr_dev) | [GitHub](https://github.com/kha-white/manga-ocr) |

## Translators

| Model      | Paper                                                                                                                                                         | Train                                                  | Source                                                                                                                                                      |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| m2m100     | [arxiv](https://arxiv.org/abs/2010.11125)                                                                                                                     | [Fairseq](https://github.com/facebookresearch/fairseq) | [Hugging Face](https://huggingface.co/docs/transformers/model_doc/m2m_100) [Github](https://github.com/facebookresearch/fairseq/tree/main/examples/m2m_100) |
| mbart      | [arxiv](https://arxiv.org/abs/2001.08210)                                                                                                                     | [Fairseq](https://github.com/facebookresearch/fairseq) | [Hugging Face](https://huggingface.co/docs/transformers/model_doc/mbart) [Github](https://github.com/facebookresearch/fairseq/tree/main/examples/mbart)     |
| nllb       | [arxiv](https://arxiv.org/abs/2207.04672)                                                                                                                     | [Fairseq](https://github.com/facebookresearch/fairseq) | [Hugging Face](https://huggingface.co/docs/transformers/model_doc/nllb) [GitHub](https://github.com/gordicaleksa/Open-NLLB)                                 |
| sugoi      | /                                                                                                                                                             | [Fairseq](https://github.com/facebookresearch/fairseq) | [Blog](https://blog.sugoitoolkit.com/author/minh/) [Patreon](https://www.patreon.com/mingshiba/)                                                            |
| jparacrawl | [arxiv](https://arxiv.org/abs/2405.09017) [aclanthology](https://aclanthology.org/2022.lrec-1.721/) [aclanthology](https://aclanthology.org/2020.lrec-1.443/) | [Fairseq](https://github.com/facebookresearch/fairseq) | [HomePage](https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/)                                                                                                 |
| qwen2      | /                                                                                                                                                             | /                                                      | [Blog](https://qwenlm.github.io/blog/qwen2.5) [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) [Github](https://github.com/QwenLM)           |

# Roadmap

- [x] detectors
  - [x] dbnet
  - [x] none
  - [x] ctd
  - [x] [paddle](https://github.com/mg-chao/paddle-ocr-rs)
  - [x] dbnet_convnext
  - [ ] ~~craft~~
- [ ] ocr
  - [ ] 32px
  - [ ] 48px
  - [ ] 48px_ctc
  - [ ] mocr
    - [x] greedy
    - [ ] beam
- [ ] inpainter
  - [ ] default
  - [ ] lama_large
  - [ ] lama_mpe
  - [ ] sd
  - [ ] none
  - [ ] original
- [ ] colorizer
  - [ ] none
  - [ ] mc2
- [ ] renderer
  - [ ] json/struct
  - [ ] gimp
  - [ ] [psd](https://crates.io/crates/psd)
  - [ ] gimp
  - [ ] html
  - [ ] png
- [ ] upscaler python integration
- [ ] translator python integration
  - [~] baidu
  - [~] caiyun
  - [~] google
  - [~] m2m100
  - [~] mbart
  - [~] nllb
  - [~] none
  - [~] original
  - [~] papgo
  - [~] qwen2
  - [~] sugoi
  - [~] jparacrawl
  - [~] youdao
  - [ ] chatgpt
  - [ ] groq
  - [ ] deepl
  - [ ] deepseek
  - [ ] gemini
  - [ ] sakrua
- [ ] cleanup code
- [ ] more tests(100% test coverage)
- [ ] more benchmarks
- [ ] optimize code
- [~] error handling
- [ ] replace clipper2
- [ ] replace opencv
- [x] ci
  - [x] cargo build
  - [x] gh publish
  - [x] cargo test
  - [x] cargo fmt
  - [ ] cargo clippy
  - [ ] cargo doc
  - [ ] cargo tarpaulin
  - [x] pyo3 publish
    - [x] macos arm64
    - [x] macos x86_64
    - [x] linux x86_64
    - [x] linux arm64
    - [x] windows x86_64
    - [ ] windows arm64(no prebuild clang)
    - [ ] ~~windows x86~~

# Deploy

- opencv
- onnxruntime exectuion providers
- main binary

# CPP Dependencies

- [opencv](https://docs.opencv.org/4.x/index.html)
- [ort/onnxruntime](https://github.com/microsoft/onnxruntime)
- [clipper2](https://github.com/AngusJohnson/Clipper2)
