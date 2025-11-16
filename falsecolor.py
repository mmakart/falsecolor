from PIL import Image
import falsecolor
import os.path
import math
import numpy as np
import time
import sys

TILE_SIZE = 16

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
    return round(p * 255.0)

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

    # May go beyond range 0-255 for RGB components
    r = to_int((to_float(dst[0]) - to_float(src[0]) * alpha) / (1.0 - alpha))
    g = to_int((to_float(dst[1]) - to_float(src[1]) * alpha) / (1.0 - alpha))
    b = to_int((to_float(dst[2]) - to_float(src[2]) * alpha) / (1.0 - alpha))

    # ... because they're clipped to range 0-255 inside
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

def calc_smudges(target, canvas, error_tolerance):
    opt_start = time.time()
    steps = falsecolor.fit(np.array(target), np.array(canvas), error_tolerance)
    opt_time = time.time() - opt_start

    print(f'fitting took {opt_time:.4} seconds')

    return steps

def apply_all(image, steps):
    brush_operations = {'p': paint_1px, 'w': smudge_water, 'o': smudge_oil}

    copy = image.copy()
    for x, y, brush, brush_type in steps:
        try:
            brush_operations[brush_type](copy, x, y, brush)
        except KeyError:
            raise ValueError(f'Unknown brush type: {brush_type}')

    return copy

def save_instructions_txt(steps, filename):

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as fout:
        offsets = {'w': '', 'p': ' ', 'o': '  '}
        hints = calc_hotbar_exchange_hints(steps)

        annotation='''\
# Legend:
# x: column [1..width]
# y: row [1..height] (from top to bottom)
# t: brush type:
#     p: 1 pixel brush
#     w: watercolor brush
#     o: oil brush

#  #:   x  y  t color
'''
        fout.write(annotation)

        for i, (x, y, brush, brush_type) in enumerate(steps):
            if i == 0:
                current_x, current_y = x, y
            elif not (x == current_x and y == current_y):
                dx, dy = x - current_x, y - current_y
                horizontal_move = f'> {dx}' if dx > 0 else (f'< {-dx}' if dx < 0 else '')
                vertical_move = f'v {dy}' if dy > 0 else (f'^ {-dy}' if dy < 0 else '')

                fout.write('\n#    ' + f'{horizontal_move} {vertical_move}'.strip() + '\n')

                current_x, current_y = x, y

            smudge = f'{i+1:5}:  ({x+1:2} {y+1:2})  {offsets[brush_type]}{brush_type} {brush:10}'
            smudge += f' - {hints[i][1]} {hints[i][0]}\n' if i in hints else '\n'
            fout.write(smudge)

def save_intermediate_images(image, steps, apply_per_brush, tile_pos, base_directory):
    xtile, ytile = tile_pos

    current_dir = os.path.join(base_directory, f'row{ytile + 1}_column{xtile + 1}')
    os.makedirs(current_dir)

    copy = image.copy()

    copy.save(os.path.join(current_dir, f'{0:07}.png'))
    for i, (x, y, brush, brush_type) in enumerate(steps):
        try:
            apply_per_brush[brush_type](copy, x, y, brush)
        except KeyError:
            raise ValueError(f'Unknown brush type: {brush_type}')

        copy.save(os.path.join(current_dir, f'{i + 1:07}.png'))

    return copy

def make_tiled_image(target, initial_image, tile_size, error_tolerance, output_dir, save_debug_img):
    x_tiles = math.ceil(target.width / tile_size)
    y_tiles = math.ceil(target.height / tile_size)

    print(f'tiling {x_tiles}x{y_tiles}')

    total_steps = 0

    result_image = Image.new(initial_image.mode, initial_image.size)

    for xtile in range(0, x_tiles):
        for ytile in range(0, y_tiles):
            crop_coords = (xtile * tile_size, ytile * tile_size,
                    xtile * tile_size + tile_size, ytile * tile_size + tile_size)
            print(crop_coords)

            target_tile = target.crop(crop_coords)
            canvas_tile = initial_image.crop(crop_coords)

            steps = calc_smudges(target_tile, canvas_tile, error_tolerance)

            result_tile = apply_all(canvas_tile, steps)
            result_image.paste(result_tile, crop_coords[:2])

            total_steps += len(steps)

            save_instructions_txt(steps, os.path.join(output_dir,
                    'instructions', f'row{ytile + 1}_column{xtile + 1}.txt'))

            if save_debug_img:
                apply_per_brush = {'p': paint_1px, 'w': smudge_water, 'o': smudge_oil}
                rev_apply_per_brush = {'p': paint_1px_reverse,
                        'w': smudge_water_reverse, 'o': smudge_oil_reverse}

                save_intermediate_images(canvas_tile, steps, apply_per_brush,
                        (xtile, ytile), os.path.join(output_dir, 'hist'))
                after_reversed = save_intermediate_images(target_tile,
                        reversed(steps), rev_apply_per_brush, (xtile, ytile),
                        os.path.join(output_dir, 'rev'))
                save_intermediate_images(after_reversed, steps,
                        apply_per_brush, (xtile, ytile),
                        os.path.join(output_dir, 'after_rev'))

    print(f'    Total smudges in all canvases: {total_steps}')

    result_image.save(sys.argv[2])

def calc_hotbar_exchange_hints(steps):
    HOTBAR_CAPACITY = 8

    all_indexes = {}

    for i, (x, y, color, brush_type) in enumerate(steps):
        item = (color, brush_type)
        if item in all_indexes:
            all_indexes[item].append(i)
        else:
            all_indexes[item] = [i]

    for item in all_indexes.keys():
        all_indexes[item].append(float('inf'))

    hints = {}
    hotbar = set()
    current_indexes = {item: 0 for item in all_indexes.keys()}

    for i, (x, y, color, brush_type) in enumerate(steps):
        item = (color, brush_type)

        if item not in hotbar:
            if len(hotbar) >= HOTBAR_CAPACITY:
                furthest_item = max(((item, all_indexes[item][current_indexes[item]])
                        for item in hotbar), key=lambda el: el[1])[0]

                hotbar.remove(furthest_item)
                hints[i] = furthest_item

            hotbar.add(item)

        current_indexes[item] += 1

    return hints




def main():
    save_debug_img = '--save-intermediate' in sys.argv

    target = Image.open(sys.argv[1]).convert('RGB')

    print(f'target.size {target.size}')

    error_tolerance = float(sys.argv[3])

    if len(sys.argv) >= 5 and sys.argv[4] != '--save-intermediate': # TODO: dirty
        initial_image = Image.open(sys.argv[4]).convert('RGB')
        if initial_image.size != target.size:
            raise ValueError('Canvas and target must have equal width and height')
    else:
        initial_image = Image.new(target.mode, target.size, (255, 255, 255))

    # output image name with extention removed (i.e. 'nature' for 'my/dir/name/nature.png')
    output_dir = os.path.splitext(os.path.split(sys.argv[2])[-1])[0]

    make_tiled_image(target, initial_image, TILE_SIZE, error_tolerance, output_dir, save_debug_img)

if __name__ == '__main__':
    main()
