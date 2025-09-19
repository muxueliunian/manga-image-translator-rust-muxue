# Python Renderer Usage

```sh
❯ cargo r -p simple-runtime -- -i path/to/input -o path/to/output
❯ ./runtime -i path/to/input -o path/to/output

Options:
  -i, --input <INPUT>    Input file or directory
  -o, --output <OUTPUT>  Output directory
  -c, --config <CONFIG>  Optional config file
  -v, --verbose...       Verbose mode (-v, -vv, -vvv)
      --overwrite        Overwrite already translated images
  -h, --help             Print help
  -V, --version          Print version
```

```sh
❯ ./scripts/python-render.py -i path/to/input.mit.bin -o path/to/output.png
usage: python-render.py [-h] -i INPUT -o OUTPUT
                        [--renderer {Renderer.default,Renderer.manga2Eng,Renderer.manga2EngPillow}]
                        [--font-path FONT_PATH] [--line_spacing LINE_SPACING] [--no_hyphenation]
                        [--font_size FONT_SIZE] [--font_size_offset FONT_SIZE_OFFSET]
                        [--font_size_minimum FONT_SIZE_MINIMUM]
```
