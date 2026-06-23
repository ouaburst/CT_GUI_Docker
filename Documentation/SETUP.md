# How to generate ConeBeamGeometry data

These are the data files and their shapes.

- sinogram.npy - The sinogram data. Shape: `(N, X, Y)`
- angles.npy - The source/detector angles. Shape: `(N,)`
- axial_position.npy - The z axis position of the source/detector. Shape: `(N,)`
- shifts.npy - Fine scale shifts of the source/detector position. Shape: `(N, 3)`
- metadata.json - Describes the geometry setup.

The variables in the data file shapes have the following meaning:
- `N`: The number of projections/images.
- `X`: The number of sensor positions in the X axis of the sensor.
- `Y`: The number of sensor positions in the Y axis of the sensor.


```
Expected layout on host (bind-mount into the container at the SAME path):
<samples_directory>/<specie>_<tree_ID>_<disk_ID>/
    ├─ sinogram.npy
    ├─ angles.npy
    ├─ axial_positions.npy
    ├─ shifts.npy
    └─ metadata.json
```

Run `docker run --rm -it -p 8000:8000  --gpus all  -v /media/Store-SSD:/media/Store-SSD:ro -v /home/julius/CT_GUI_Docker/test:/test  woodscan:cuda121  python generate_sinogram.py`


# Limitations

- Currently only supports odl `ConeBeamGeometry` geometry.
- Only tested with a curved sensor.
- Only support square sensor pixels
- Currently no translation and no axial shift
- Sample ID scheme is hardcoded.