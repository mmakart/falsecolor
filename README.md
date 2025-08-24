```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ..
python falsecolor.py input.png output.png {error_tolerance} [canvas.png] [--save-intermediate]
```

`error_tolerance`: valid range: (0; 1]. The lower - the better quality.

Result image will be `output.png`. Painting instructions will be `instructions.txt`.

If you have not default (white, blank) canvas to start with, you can provide optional filename for it (`canvas.png` in the example above).

If `--save-intermediate` option is provided you should create `hist`, `rev` and `after_rev` folders before running otherwise the program will crash.
