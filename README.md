```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ..
python falsecolor.py input.png output.png {error_tolerance} [--save-intermediate]
```

`error_tolerance`: valid range: (0; 1]. The lower - the better quality.

Result image will be `output.png`. Drawing instructions will be `instructions.txt`.

If `--save-intermediate` option is provided you should create `hist`, `rev` and `after_rev` folders before running otherwise the program would crash.
