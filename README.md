# Falsecolor

## Compiling

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Running

```
python falsecolor.py {input.png} {output.png} {error_tolerance} [canvas.png] [--save-intermediate]
```

## Output

Result image will be `output.png`.

The `.txt` files with painting instructions will be in `output` folder (`path/to/output.png` -> `output.png` -> `output`).

If `--save-intermediate` argument is provided, 3 additional folders in the output directory will be created `hist`, `rev` and `after_rev`.

* If you want to visually inspect the painting process, you only need images in `hist`.
* `rev` and `after_rev` folders contain images intended for debug purposes.

## Arguments

### Mandatory

* `input.png`: path to reference image.
* `output.png`: path to result image.
* `error_tolerance`: valid range: (0; 1]. The lower - the better quality.

### Optional

* `canvas.png`: if you have custom canvas to start with, provide this image.
* `--save-intermediate`: see "Output".
