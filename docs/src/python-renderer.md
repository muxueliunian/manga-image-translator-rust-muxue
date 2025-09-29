# Python Renderer Usage
The runtime allows to export the processed image, before the text is rendered. This output can be used with the original Renderer from the python project. 

After running the runtime you can run the Python renderer script.

## Install
```sh
# Setup virtual environment
python3 -m venv venv && source venv/bin/activate

# Install dependencies
pip install numpy Pillow git+https://github.com/frederik-uni/manga-image-translator.git@renderer-module#subdirectory=pip-modules/mit-renderer

# Install Python renderer script
curl -O https://raw.githubusercontent.com/frederik-uni/manga-image-translator-rust/master/scripts/python-render.py

# Download fonts
REPO="zyddnys/manga-image-translator"; FOLDER="fonts"; BRANCH="main"; mkdir -p "$FOLDER"; curl -s "https://api.github.com/repos/$REPO/contents/$FOLDER?ref=$BRANCH" | jq -r '.[] | select(.type=="file") | .download_url' | while read -r url; do fname=$(basename "$url"); fname=$(python3 -c "import urllib.parse; print(urllib.parse.unquote('$fname'))"); curl -L "$url" -o "$FOLDER/$fname"; done

```

## Usage
```sh
❯ ./python-render.py -i path/to/input.mit.bin -o path/to/output.png
usage: python-render.py [-h] -i INPUT -o OUTPUT
                        [--renderer {Renderer.default,Renderer.manga2Eng,Renderer.manga2EngPillow}]
                        [--font-path FONT_PATH] [--line_spacing LINE_SPACING] [--no_hyphenation]
                        [--font_size FONT_SIZE] [--font_size_offset FONT_SIZE_OFFSET]
                        [--font_size_minimum FONT_SIZE_MINIMUM]
```
