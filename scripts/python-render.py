#!/usr/bin/env python3

import struct
from enum import Enum

try:
    import mit_renderer
    import PIL
    import numpy as np

except ImportError:
    print("(Optional) Use venv: `python3 -m venv venv && source venv/bin/activate`")
    print(
        "install modules with: `pip install numpy Pillow git+https://github.com/frederik-uni/manga-image-translator.git@renderer-module#subdirectory=pip-modules/mit-renderer`"
    )
    exit(1)


def load_map(buffer: bytes, start_offset: int = 0) -> tuple[dict[str, str], int]:
    translations = {}
    offset = start_offset

    # Number of entries (u64 little-endian)
    (num_entries,) = struct.unpack_from("<Q", buffer, offset)
    offset += 8

    for _ in range(num_entries):
        (key_len,) = struct.unpack_from("<Q", buffer, offset)
        offset += 8
        key = buffer[offset : offset + key_len].decode("utf-8")
        offset += key_len

        (value_len,) = struct.unpack_from("<Q", buffer, offset)
        offset += 8
        value = buffer[offset : offset + value_len].decode("utf-8")
        offset += value_len

        translations[key] = value

    return translations, offset - start_offset


class RustTextBlock:
    fg_color: tuple[int, ...]
    bg_color: tuple[int, ...]

    def __init__(self):
        self.font_size = 0
        self.angle = 0.0
        self.prob = 0.0
        self.skip_translate = False
        self.fg_color = (0, 0, 0)
        self.bg_color = (0, 0, 0)
        self.text = ""
        self.lines = None  # numpy array of shape (num_lines, 4, 2), dtype=int64
        self.translations = {}

    def __repr__(self):
        return f"RustTextBlock<{self.angle} degrees | {self.prob} | {self.text} >"


def load_textblock(data: bytes) -> RustTextBlock:
    tb = RustTextBlock()
    offset = 0

    (tb.font_size,) = struct.unpack_from("<Q", data, offset)
    offset += 8

    (tb.angle,) = struct.unpack_from("<d", data, offset)
    offset += 8

    (tb.prob,) = struct.unpack_from("<d", data, offset)
    offset += 8

    tb.skip_translate = bool(data[offset])
    offset += 1

    if data[offset]:
        tb.fg_color = tuple(data[offset + 1 : offset + 4])
        offset += 4
    else:
        offset += 1

    if data[offset]:
        tb.bg_color = tuple(data[offset + 1 : offset + 4])
        offset += 4
    else:
        offset += 1

    (text_len,) = struct.unpack_from("<Q", data, offset)
    offset += 8
    tb.text = data[offset : offset + text_len].decode("utf-8")
    offset += text_len

    (num_lines,) = struct.unpack_from("<Q", data, offset)
    offset += 8

    lines = []
    for _ in range(num_lines):
        line = []
        for _ in range(4):
            (x,) = struct.unpack_from("<q", data, offset)  # i64
            offset += 8
            (y,) = struct.unpack_from("<q", data, offset)
            offset += 8
            line.append([x, y])
        lines.append(line)

    tb.lines = np.array(lines, dtype=np.int64)
    trans, new_offset = load_map(data[offset:])
    tb.translations = trans
    offset += new_offset
    return tb, offset


class Image:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.raw = False
        self.data = None

    def __repr__(self):
        data_len = self.data.nbytes if self.data is not None else 0
        return f"Image<{self.width}x{self.height} | {'raw' if self.raw else 'not raw'} | {data_len} bytes>"


def load_image(data: bytes) -> Image:
    img = Image()
    offset = 0

    (img.width,) = struct.unpack_from("<H", data, offset)
    offset += 2
    (img.height,) = struct.unpack_from("<H", data, offset)
    offset += 2

    img.raw = bool(data[offset])
    offset += 1

    (data_len,) = struct.unpack_from("<Q", data, offset)
    offset += 8

    raw_data = data[offset : offset + data_len]

    if img.raw:
        total_pixels = img.height * img.width
        total_values = len(raw_data)

        channels = total_values // total_pixels
        img.data = (
            np.frombuffer(raw_data, dtype=np.uint8)
            .reshape((img.height, img.width, channels))
            .copy()
        )
    else:
        img.data = raw_data
    offset += data_len
    return img, offset


class Patch:
    info: RustTextBlock | None
    bg: Image | None

    def __init__(self):
        self.pos = (0, 0)
        self.bg = None  # Image
        self.info = None  # RustTextBlock

    def __repr__(self):
        return f"Patch<{self.pos[0]}x{self.pos[1]} | {self.bg} | {self.info} >"


def load_patch(data: bytes) -> Patch:
    patch = Patch()
    offset = 0

    (x,) = struct.unpack_from("<Q", data, offset)
    offset += 8
    (y,) = struct.unpack_from("<Q", data, offset)
    offset += 8
    patch.pos = (x, y)

    patch.bg, total_bg_bytes = load_image(data[offset:])
    offset += total_bg_bytes

    patch.info, new_offset = load_textblock(data[offset:])
    offset += new_offset
    return patch, offset


class Export:
    img: Image

    def __init__(self):
        self.img = None  # Image
        self.patches = []


def load_export(data: bytes) -> Export:
    export = Export()

    export.img, offset = load_image(data)

    (num_patches,) = struct.unpack_from("<Q", data, offset)
    offset += 8

    patches = []
    for _ in range(num_patches):
        patch, new_offset = load_patch(data[offset:])
        offset += new_offset
        patches.append(patch)
    export.patches = patches

    return export


class Renderer(str, Enum):
    default = "default"
    manga2Eng = "manga2eng"
    manga2EngPillow = "manga2eng_pillow"


async def run_render(
    renderer,
    text_regions,
    rgb_img,
    img_inpainted,
    font_path,
    line_spacing,
    no_hyphenation,
    font_size,
    font_size_offset,
    font_size_minimum,
):
    from mit_renderer import (
        dispatch as dispatch_rendering,
        dispatch_eng_render,
        dispatch_eng_render_pillow,
    )

    if (
        renderer == Renderer.manga2Eng or renderer == Renderer.manga2EngPillow
    ) and text_regions:
        if renderer == Renderer.manga2EngPillow:
            output = await dispatch_eng_render_pillow(
                img_inpainted,
                rgb_img,
                text_regions,
                font_path,
                line_spacing,
            )
        else:
            output = await dispatch_eng_render(
                img_inpainted,
                rgb_img,
                text_regions,
                font_path,
                line_spacing,
            )
    else:
        output = await dispatch_rendering(
            img_inpainted,
            text_regions,
            font_path,
            font_size,
            font_size_offset,
            font_size_minimum,
            not no_hyphenation,
            None,
            line_spacing,
        )
    return output


async def main():
    import argparse
    from pathlib import Path
    import logging
    from urllib.parse import unquote
    import os

    def url_decode(s):
        s = unquote(s)
        if s.startswith("file:///"):
            s = s[len("file://") :]
        return s

    def file_path(string):
        if not string:
            return ""
        s = url_decode(os.path.expanduser(string))
        if not os.path.exists(s):
            raise argparse.ArgumentTypeError(f'No such file: "{string}"')
        return s

    parser = argparse.ArgumentParser(
        description="Load and log an Export from a binary file."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to the binary file to load",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Filepath/filename to save the rendered image to",
    )
    parser.add_argument(
        "--renderer",
        type=Renderer,
        default=Renderer.manga2EngPillow,
        choices=list(Renderer),
        help="Select the renderer.",
    )
    parser.add_argument(
        "--font-path", default="", type=file_path, help="Path to font file"
    )
    parser.add_argument(
        "--line_spacing", type=int, default=None, help="Line spacing in pixels."
    )
    parser.add_argument(
        "--no_hyphenation", action="store_true", help="Disable hyphenation."
    )
    parser.add_argument(
        "--font_size", type=int, default=None, help="Font size in points."
    )
    parser.add_argument(
        "--font_size_offset", type=int, default=0, help="Offset to apply to font size."
    )
    parser.add_argument(
        "--font_size_minimum", type=int, default=-1, help="Minimum font size allowed."
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    # Load the file
    if not args.input.exists():
        logging.error(f"File {args.input} does not exist!")
        return

    with open(args.input, "rb") as f:
        data = f.read()

    export = load_export(data)
    img = export.img.data
    orig_rgb_img = img.copy()[..., :3]
    blocks = []
    for patch in export.patches:
        patch: Patch
        x = patch.pos[0]
        y = patch.pos[1]
        patch_h, patch_w = patch.bg.height, patch.bg.width

        ph = min(patch_h, img.shape[0] - y)
        pw = min(patch_w, img.shape[1] - x)
        mask = patch.bg.data[:ph, :pw, 3:4] >= 128
        rgb_channels = img[..., :3]
        rgb_channels[y : y + ph, x : x + pw] = patch.bg.data[:ph, :pw, :3] * mask + img[
            y : y + ph, x : x + pw
        ] * (1 - mask)
        fg_color = tuple(c / 255 for c in (patch.info.fg_color or (0, 0, 0)))
        bg_color = tuple(c / 255 for c in (patch.info.bg_color or (255, 255, 255)))
        block = mit_renderer.TextBlock(
            patch.info.lines,
            [patch.info.text],
            translation=patch.info.translations.get("translated"),
            font_size=patch.info.font_size,
            angle=patch.info.angle,
            prob=patch.info.lines,
            fg_color=fg_color,
            bg_color=bg_color,
        )
        blocks.append(block)

    img_inpainted = img[..., :3]
    img = await run_render(
        args.renderer,
        blocks,
        orig_rgb_img,
        img_inpainted,
        args.font_path,
        args.line_spacing,
        args.no_hyphenation,
        args.font_size,
        args.font_size_offset,
        args.font_size_minimum,
    )

    image = PIL.Image.fromarray(img)

    image.save(args.output)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
