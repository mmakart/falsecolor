from PIL import Image
import numpy as np
import falsecolor
import time
import sys

save_debug_img = '--save-intermediate' in sys.argv

target = Image.open(sys.argv[1]).convert('RGB')
error_tolerance = float(sys.argv[3])

if len(sys.argv) >= 5 and sys.argv[4] != '--save-intermediate': # TODO: dirty
    tiled_canvas = Image.open(sys.argv[4]).convert('RGB')
    if tiled_canvas.size != target.size:
        raise ValueError('Canvas and target must have equal width and height')
else:
    tiled_canvas = Image.new(target.mode, (target.width, target.height), (255, 255, 255))

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
    return int(round(p * 255.0))

def blend(img: Image.Image, pos, src, alpha):
    if pos[0] >= img.size[0] or pos[1] >= img.size[1] or pos[0] < 0 or pos[1] < 0:
        return

    dst = img.getpixel(pos)

    r = to_int(to_float(dst[0]) * (1.0 - alpha) + to_float(src[0]) * alpha)
    g = to_int(to_float(dst[1]) * (1.0 - alpha) + to_float(src[1]) * alpha)
    b = to_int(to_float(dst[2]) * (1.0 - alpha) + to_float(src[2]) * alpha)

    img.putpixel(pos, (r, g, b))

def blend_reverse(img: Image.Image, pos, src, alpha):
    if pos[0] >= img.size[0] or pos[1] >= img.size[1] or pos[0] < 0 or pos[1] < 0:
        return

    dst = img.getpixel(pos)

    r = to_int((to_float(dst[0]) - to_float(src[0]) * alpha) / (1.0 - alpha))
    g = to_int((to_float(dst[1]) - to_float(src[1]) * alpha) / (1.0 - alpha))
    b = to_int((to_float(dst[2]) - to_float(src[2]) * alpha) / (1.0 - alpha))

    img.putpixel(pos, (r, g, b))

def smudge_water(im, x, y, brush):
    b = brushes[brush]
    blend(im, (x, y), b, 0.7)
    blend(im, (x+1, y), b, 0.5)
    blend(im, (x, y+1), b, 0.5)
    blend(im, (x-1, y), b, 0.5)
    blend(im, (x, y-1), b, 0.5)

def smudge_water_reverse(im, x, y, brush):
    b = brushes[brush]
    blend_reverse(im, (x, y), b, 0.7)
    blend_reverse(im, (x+1, y), b, 0.5)
    blend_reverse(im, (x, y+1), b, 0.5)
    blend_reverse(im, (x-1, y), b, 0.5)
    blend_reverse(im, (x, y-1), b, 0.5)

def smudge_oil(im, x, y, brush):
    b = brushes[brush]
    blend(im, (x, y), b, 1.0)
    blend(im, (x+1, y), b, 0.9)
    blend(im, (x, y+1), b, 0.9)
    blend(im, (x-1, y), b, 0.9)
    blend(im, (x, y-1), b, 0.9)

def smudge_oil_reverse(im, x, y, brush):
    b = brushes[brush]
    #blend_reverse(im, (x, y), b, 1.0) # division by zero!
    im.putpixel((x, y), (255, 255, 255))
    blend_reverse(im, (x+1, y), b, 0.9)
    blend_reverse(im, (x, y+1), b, 0.9)
    blend_reverse(im, (x-1, y), b, 0.9)
    blend_reverse(im, (x, y-1), b, 0.9)

def paint_1px(im, x, y, brush):
    im.putpixel((x, y), brushes[brush])

def paint_1px_reverse(im, x, y, brush):
    im.putpixel((x, y), (255, 255, 255))

def do_16x16(target, canvas, error_tolerance):
    target_data = np.array(target)
    canvas_data = np.array(canvas)

    opt_start = time.time()
    steps = falsecolor.fit(target_data, canvas_data, error_tolerance)
    opt_time = time.time() - opt_start

    print(f'fitting took {opt_time:.4} seconds')

    if save_debug_img:
        save_hist(canvas, steps)
        after_reversed = save_reversed_steps(target, steps)
        save_after_reversed_steps(after_reversed, steps)

    apply_all(canvas, steps)
    save_instructions_txt(steps, 'instructions.txt')

    return canvas, len(steps)

def apply_all(canvas, steps):
    for x, y, brush, brush_type in steps:
        if brush_type == 'p':
            paint_1px(canvas, x, y, brush)
        elif brush_type == 'w':
            smudge_water(canvas, x, y, brush)
        elif brush_type == 'o':
            smudge_oil(canvas, x, y, brush)
        else:
            raise ValueError(f'Unknown brush type: {brush_type}')

def save_instructions_txt(steps, filename):
    offsets = {'w': 0, 'p': 1, 'o': 2}

    with open(filename, 'w') as fout:
        annotation='''\
# Legend:
# x: column [1..width]
# y: row [1..height] (from top to bottom)
# t: brush type:
#     p: 1 pixel brush
#     w: watercolor brush
#     w: oil brush

#  #:  x  y t color
'''
        fout.write(annotation)

        for i, (x, y, brush, brush_type) in enumerate(steps):
            if i == 0:
                current_x, current_y = x, y
            elif not (x == current_x and y == current_y):
                dx, dy = x - current_x, y - current_y
                horizontal_move = f'> {dx}' if dx > 0 else (f'< {-dx}' if dx < 0 else '')
                vertical_move = f'v {dy}' if dy > 0 else (f'^ {-dy}' if dy < 0 else '')

                fout.write('\n#   ' + f'{horizontal_move} {vertical_move}'.strip() + '\n')

                current_x, current_y = x, y

            fout.write(f'{i+1:4}: {x+1:2} {y+1:2} {" " * offsets[brush_type]}{brush_type} {brush:10}\n')

def save_hist(canvas, steps):
    canvas_copy = canvas.copy()
    canvas_copy.save(f'hist/{0:04}.png')
    for i, (x, y, brush, brush_type) in enumerate(steps):
        if brush_type == 'p':
            paint_1px(canvas_copy, x, y, brush)
        elif brush_type == 'w':
            smudge_water(canvas_copy, x, y, brush)
        elif brush_type == 'o':
            smudge_oil(canvas_copy, x, y, brush)
        else:
            raise ValueError(f'Unknown brush type: {brush_type}')
        canvas_copy.save(f'hist/{i+1:04}.png')

def save_reversed_steps(target, steps):
    target_copy = target.copy()
    target_copy.save(f'rev/{0:04}.png')
    for i, (x, y, brush, brush_type) in enumerate(reversed(steps)):
        if brush_type == 'p':
            paint_1px_reverse(target_copy, x, y, brush)
        elif brush_type == 'w':
            smudge_water_reverse(target_copy, x, y, brush)
        elif brush_type == 'o':
            smudge_oil_reverse(target_copy, x, y, brush)
        else:
            raise ValueError(f'Unknown brush type: {brush_type}')
        target_copy.save(f'rev/{i+1:04}.png')
    return target_copy

def save_after_reversed_steps(after_reversed, steps):
    after_reversed.save(f'after_rev/{0:04}.png')
    for i, (x, y, brush, brush_type) in enumerate(steps):
        if brush_type == 'p':
            paint_1px(after_reversed, x, y, brush)
        elif brush_type == 'w':
            smudge_water(after_reversed, x, y, brush)
        elif brush_type == 'o':
            smudge_oil(after_reversed, x, y, brush)
        else:
            raise ValueError(f'Unknown brush type: {brush_type}')
        after_reversed.save(f'after_rev/{i+1:04}.png')

x_tiles = (target.size[0] + 15) // 16
y_tiles = (target.size[1] + 15) // 16

print(f'target.size {target.size}')
print(f'tiling {x_tiles}x{y_tiles}')

total_steps = 0

for xtile in range(0, x_tiles):
    for ytile in range(0, y_tiles):
        crop_coords = (xtile*16, ytile*16, xtile*16+16, ytile*16+16)
        print(crop_coords)

        target_tile = target.crop(crop_coords)
        canvas_tile = tiled_canvas.crop(crop_coords)

        tile_output, num_tile_steps = do_16x16(target_tile, canvas_tile, error_tolerance)

        tiled_canvas.paste(tile_output, (xtile*16, ytile*16))

        total_steps += num_tile_steps

print(f'Total smudges in all canvases: {total_steps}')

tiled_canvas.save(sys.argv[2])
