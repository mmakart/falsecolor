from PIL import Image
import numpy as np
import falsecolor
import time
import sys

target = Image.open(sys.argv[1]).convert('RGB')

brushes = {
    "White": (0xff, 0xff, 0xff),
    "Yellow": (0xff, 0xf0, 0x00),
    "Orange": (0xff, 0x6c, 0x00),
    "Red": (0xff, 0x00, 0x00),
    "Violet": (0x8a, 0x00, 0xff),
    "Blue": (0x00, 0x0c, 0xff),
    "Green": (0x0c, 0xff, 0x00),
    "Magenta": (0xfc, 0x00, 0xff),
    "Cyan": (0x00, 0xff, 0xea),
    "Grey": (0xbe, 0xbe, 0xbe),
    "DarkGrey": (0x7b, 0x7b, 0x7b),
    "Black": (0x00, 0x00, 0x00),
    "DarkGreen": (0x00, 0x64, 0x00),
    "Brown": (0x96, 0x4b, 0x00),
    "Pink": (0xff, 0xc0, 0xcb),
}

def to_float(p):
    return float(p) / 255.0

def to_int(p):
    return int(p * 255.0)

def blend(img: Image.Image, pos, src, alpha):
    if pos[0] >= img.size[0] or pos[1] >= img.size[1] or pos[0] < 0 or pos[1] < 0:
        return

    dst = img.getpixel(pos)

    r = to_int(to_float(dst[0]) * (1.0 - alpha) + to_float(src[0]) * alpha)
    g = to_int(to_float(dst[1]) * (1.0 - alpha) + to_float(src[1]) * alpha)
    b = to_int(to_float(dst[2]) * (1.0 - alpha) + to_float(src[2]) * alpha)

    img.putpixel(pos, (r, g, b))

def smudge(im, x, y, brush):
    b = brushes[brush]
    blend(im, (x, y), b, 0.7)
    blend(im, (x+1, y), b, 0.5)
    blend(im, (x, y+1), b, 0.5)
    blend(im, (x-1, y), b, 0.5)
    blend(im, (x, y-1), b, 0.5)

def do_16x16(input):
    output = Image.new(input.mode, (16, 16))
    output.paste((255, 255, 255), (0, 0, output.size[0], output.size[1]))

    target_data = np.array(input)

    opt_start = time.time()
    steps = falsecolor.fit(target_data)
    opt_time = time.time() - opt_start

    print(f'fitting took {opt_time} seconds')

    for x, y, brush in steps:
        smudge(output, x, y, brush)

    return output

tiled_canvas = Image.new(target.mode, target.size)

x_tiles = (target.size[0] + 15) // 16
y_tiles = (target.size[1] + 15) // 16

print(f'target.size {target.size}')
print(f'tiling {x_tiles}x{y_tiles}')

for xtile in range(0, x_tiles):
    for ytile in range(0, y_tiles):
        crop_rect = (xtile*16, ytile*16, xtile*16+16, ytile*16+16)
        print(crop_rect)
        tile = target.crop(crop_rect)

        tile_output = do_16x16(tile)

        tiled_canvas.paste(tile_output, (xtile*16, ytile*16))

tiled_canvas.save(sys.argv[2])
