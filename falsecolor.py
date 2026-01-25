from PIL import Image
import falsecolor
from collections import Counter
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

    # No need to worry if any of (r, g, b) go out of range 0-255
    r = to_int((to_float(dst[0]) - to_float(src[0]) * alpha) / (1.0 - alpha))
    g = to_int((to_float(dst[1]) - to_float(src[1]) * alpha) / (1.0 - alpha))
    b = to_int((to_float(dst[2]) - to_float(src[2]) * alpha) / (1.0 - alpha))

    # ... because they're clipped to range 0-255 inside putpixel
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

def save_instructions_txt(steps, filename, hotbar_capacity):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as fout:
        offsets = {'w': '', 'p': ' ', 'o': '  '}

        hints = calc_hotbar_exchange_hints(steps, hotbar_capacity)

        legend='''\
# Legend:
# x: column [1..width]
# y: row    [1..height] (from top to bottom)
# t: brush type
#     p: 1 pixel brush
#     w: watercolor brush
#     o: oil brush
'''
        durabilities = {'p': 256, 'w': 57, 'o': 56}
        orderings = {'w': 0, 'p': 1, 'o': 2}

        counts = Counter((c, t) for (_, _, c, t) in steps)
        items_spent = {(c, t): count / durabilities[t] for ((c, t), count) in counts.items()}

        used_brushes = '# Used brushes:\n' + '\n'.join(f'# {t + ' ' + c:12} {spent:6.3} items'
                for ((c, t), spent) in sorted(items_spent.items(),
                key=lambda el: (orderings[el[0][1]], -el[1], el[0][0]))) + '\n'
        header = '#   #:  ( x  y)  t color\n'

        fout.write('\n'.join([legend, used_brushes, header]))

        for i, (x, y, color, brush_type) in enumerate(steps):
            aux_info = []
            if i in hints.keys():
                to_remove = '# - ' + ', '.join(f'{t} {c}' for c, t in hints[i]['remove']) + '\n'
                to_add = '# + ' + ', '.join(f'{t} {c}' for c, t in hints[i]['add']) + '\n'
                aux_info.append(to_remove + to_add)

            if i == 0:
                last_x, last_y = x, y
            elif x != last_x or y != last_y:
                dx, dy = x - last_x, y - last_y
                horizontal_move = f'> {dx}' if dx > 0 else (f'< {-dx}' if dx < 0 else '')
                vertical_move = f'v {dy}' if dy > 0 else (f'^ {-dy}' if dy < 0 else '')
                aux_info.append('#    ' + f'{horizontal_move} {vertical_move}'.strip())

                last_x, last_y = x, y

            aux_str = '\n' + '\n'.join(aux_info) + '\n' if len(aux_info) != 0 else ''
            if len(aux_str) != 0:
                fout.write(aux_str)

            smudge = f'{i+1:5}:  ({x+1:2} {y+1:2})  {offsets[brush_type]}{brush_type} {color:10}\n'
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

def make_tiled_image(
        target,
        initial_image,
        tile_size,
        error_tolerance,
        output_dir,
        save_debug_img,
        hotbar_capacity=8):
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

            print(f'Total movement before: manhattan={total_movement(steps,
                    manhattan_distance)} custom={total_movement(steps, custom_distance)}')
            steps = minimize_movement(steps, (tile_size, tile_size), manhattan_distance)
            print(f'Total movement after: manhattan={total_movement(steps,
                    manhattan_distance)} custom={total_movement(steps, custom_distance)}')

            result_tile = apply_all(canvas_tile, steps)
            result_image.paste(result_tile, crop_coords[:2])

            total_steps += len(steps)

            save_instructions_txt(steps, os.path.join(output_dir,
                    'instructions', f'row{ytile + 1}_column{xtile + 1}.txt'),
                    hotbar_capacity)

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

def calc_hotbar_exchanges(steps, hotbar_capacity):
    unique_items = {(color, brush_type) for (_, _, color, brush_type) in steps}

    all_indexes = {item: [] for item in unique_items}
    for i, (_, _, color, brush_type) in enumerate(steps):
        item = (color, brush_type)
        all_indexes[item].append(i)
    for indexes in all_indexes.values():
        indexes.append(float('inf'))

    deltas_back = []
    current_deltas = {item: float('inf') for item in unique_items}
    for _, _, color, brush_type in steps:
        for item in current_deltas.keys():
            current_deltas[item] += 1

        current_item = (color, brush_type)
        current_deltas[current_item] = 0
        deltas_back.append(current_deltas.copy())

    exchanges = []
    hotbar = set()
    current_indexes = {item: 0 for item in unique_items}
    for i, (_, _, color, brush_type) in enumerate(steps):
        item = (color, brush_type)
        hotbar.add(item)
        if len(hotbar) > hotbar_capacity:
            furthest_item = max(((item, all_indexes[item][current_indexes[item]])
                    for item in hotbar), key=lambda el: (el[1], el[0][0], el[0][1]))[0]
            hotbar.remove(furthest_item)

            min_step = i - deltas_back[i][furthest_item] + 1
            max_step = i # Inclusive
            exchanges.append({'min_step': min_step, 'max_step': max_step,
                    'add': item, 'remove': furthest_item})

        current_indexes[item] += 1

    return exchanges

def group_hotbar_exchanges(exchanges):
    endpoints = []
    for i, exchange in enumerate(exchanges):
        endpoints.append((exchange['min_step'], i, 'min'))
        endpoints.append((exchange['max_step'], i, 'max'))
    endpoints.sort(key=lambda el: (el[0], 0 if el[2] == 'min' else 1))

    grouped_exchanges = {}
    current_group = {}
    visited = set()
    used_exchanges = set()
    for step, exchange_index, point_type in reversed(endpoints):
        exchange = exchanges[exchange_index]
        if point_type == 'max':
            current_group.setdefault('add', []).append(exchange['add'])
            current_group.setdefault('remove', []).append(exchange['remove'])
            used_exchanges.add(exchange_index)
        elif point_type == 'min' and exchange_index not in visited:
            visited.update(used_exchanges)
            grouped_exchanges[step] = current_group.copy()
            current_group.clear()
            used_exchanges.clear()

    return grouped_exchanges

def calc_hotbar_exchange_hints(steps, hotbar_capacity):
    exchanges = calc_hotbar_exchanges(steps, hotbar_capacity)
    return group_hotbar_exchanges(exchanges)

def get_dependencies(steps, sizes):
    width, height = sizes

    dcoords = {
        'p': (np.array([0]), np.array([0])),
        'w': (np.array([-1, 0, 0, 0, 1]), np.array([0, -1, 0, 1, 0])),
        'o': (np.array([-1, 0, 0, 0, 1]), np.array([0, -1, 0, 1, 0]))
    }

    EMPTY = np.int32(-1)

    last_steps = np.full((height, width), EMPTY, dtype=np.int32)
    steps_over = {}
    steps_under = {}

    for i, (x, y, _, brush_type) in enumerate(steps):
        try:
            coords = (dcoords[brush_type][0] + y, dcoords[brush_type][1] + x)
        except KeyError:
            raise ValueError(f'Unknown brush type: {brush_type}')

        in_bounds = ((coords[0] >= 0) & (coords[0] < height) &
                (coords[1] >= 0) & (coords[1] < width))
        coords = (coords[0][in_bounds], coords[1][in_bounds])

        covered_steps = np.unique(last_steps[coords])
        covered_steps = covered_steps[covered_steps != EMPTY]

        for covered_step in covered_steps:
            steps_over.setdefault(covered_step, set()).add(i)
            steps_under.setdefault(i, set()).add(covered_step)

        last_steps[coords] = i

    return steps_over, steps_under

def minimize_movement(steps, sizes, distance_func, chunk_size=16):
    if len(steps) == 0:
        return steps

    steps_over, steps_under = get_dependencies(steps, sizes)

    candidates = {i for i in range(len(steps)) if i not in steps_under.keys()}

    # TODO: transform to priority queue by distance if possible.
    distances = {i: {j: distance_func(steps[i][0:2], steps[j][0:2])
            for j in candidates if j != i} for i in candidates}

    # Closest to top-left corner. But it's okay to choose any other smudge.
    current_step = min(((i, steps[i][0] + steps[i][1]) for i in candidates),
            key=lambda el: el[1])[0]

    width = sizes[0]
    chunk_of_step = [((y // chunk_size) * width + (x // chunk_size))
            for (x, y, _, _) in steps]

    result = [steps[current_step]]

    # Custom greedy travelling salesman problem algorithm.
    # There aren't general applicable TSP solvers here because smudges aren't
    # directed graph because they have too complex order (a smudge can't have
    # place before any smudge it covers).
    while True:
        if current_step in steps_over.keys(): # Has smudges above
            for step_over in steps_over[current_step]:
                steps_under[step_over].remove(current_step)
                if len(steps_under[step_over]) == 0:
                    # Unblock new smudge
                    for candidate in candidates:
                        dist = distance_func(steps[step_over][0:2], steps[candidate][0:2])
                        distances.setdefault(step_over, {})[candidate] = dist
                        distances[candidate][step_over] = dist
                    candidates.add(step_over)

        candidates.remove(current_step)

        if len(candidates) == 0:
            break

        next_step = min(((other, dist) for (other, dist)
                in distances[current_step].items()),
                key=lambda el: (chunk_of_step[el[0]], el[1]))[0]
        del distances[current_step]
        for others in distances.values():
            # No-throw variant of "del others[current_step]"
            others.pop(current_step, None)

        current_step = next_step

        result.append(steps[current_step])

    return result

def manhattan_distance(p1, p2):
    return abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])

def euclidean_distance(p1, p2):
    return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[0]) ** 2) ** 0.5

def custom_distance(p1, p2):
    dist = manhattan_distance(p1, p2)
    return dist if dist == 0 else (dist + 5 if dist <= 8 else 25)

def total_movement(steps, dist_func):
    return sum(dist_func(s1[0:2], s2[0:2]) for s1, s2 in zip(steps[:-1], steps[1:]))

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
        initial_image = Image.new('RGB', target.size, (255, 255, 255))

    # output image name with extention removed (i.e. 'nature' for 'my/dir/name/nature.png')
    output_dir = os.path.splitext(os.path.split(sys.argv[2])[-1])[0]

    make_tiled_image(target, initial_image, TILE_SIZE, error_tolerance, output_dir, save_debug_img)

if __name__ == '__main__':
    main()
