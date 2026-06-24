# Data format

Sinogram and source/detector geometry data is stored in a series of numpy `.npy` array files.
These files describe the angles and position of the source and detector as well as the sinogram data itself.

For a hypothetical dataset with `N` projections and a sensor with `X` times `Y` pixels the dataset would look like the following:

- `sinogram.npy` - shape: `(N, X, Y)`

  The sinogram data itself. For each angle a 2D projection on the sensor at that angle.

- `angles.npy` - shape: `(N,)`

  The angle of the source/detector for each projection. numpy.unwrap will be called in the data so storing the angles (mod 2π) is fine.

- `axial_position.npy` - shape: `(N,)`

  The axial position (distance along the axis of rotation) of the source/detector for each projection.

- `shifts.npy` - shape: `(N, 3)`

  Fine grained corrections on the angle + z position data. A 3D vector for each projection where the values have the following meaning:

  - `shifts.npy[:, 0]`: unused
  - `shifts.npy[:, 1]`: unused
  - `shifts.npy[:, 2]`: shift along the axis of rotation

- `metadata.json` - schema: [metadata-schema.json](./metadata-schema.json)
  
  Metadata about the acquisition geometry such as the source radius and detector shape.

These files are placed in a folder that contains data for a single scan.
Individual scans are placed in a shared paren folder.

These samples are then enumerated in a `sample_config.json` file.
This file specifies the `/samples_directory` folder as well as the individual samples.

FIXME: Document format for sample_config.json.

The result is this expected layout on the server:
```
/samples_directory/
  ├─ samples_config.json
  └─ <specie>_<tree_ID>_<disk_ID>/
    ├─ sinogram.npy
    ├─ angles.npy
    ├─ axial_positions.npy
    ├─ shifts.npy
    └─ metadata.json
```

By mounting different folder into the docker container as `/samples_directory` it is possible to switch between different datasets.

# Generating example data

To be able to evaluate and test this software without having to convert real sinogram data into the format described above a script for generating example data is provided.
The script uses a phantom and calculates a series of projections of this phantom to use as example data.

[generate_sinogram.py](../generate_sinogram.py)

To run the script you need to have properly setup the server docker container as described in [README.md](../README.md).
Once you can run commands in the docker container we will run the followng command to generate example data

```
python generate_sinogram.py /samples_directory
```

The full docker command might look something like this:
```
docker run --rm -it --gpus all -v </path/to/store/example/data>:/samples_directory  woodscan:cuda121  python generate_sinogram.py /samples_directory
```

We mount some path `</path/to/store/example/data>` to `/samples_directory` in the container and then we tell `generate_sinogram.py` to write the example data to that folder.

The script will generate the example sinogram data and write it to `/samples_directory/phantom_1_1`, the script will also write out a `/samples_directory/samples_config.json` that lists all of the samples in the samples folder.

To run the server with the generated data we now just need to run the server and mount the `</path/to/store/example/data>`, where the example data is stored, as `/samples_directory`.
The full command might look something like this:
```
docker run --rm -it -p 8000:8000  --gpus all  -v </path/to/store/example/data>:/samples_directory:ro  woodscan:cuda121  python -m uvicorn odl_stream_server:app --host 0.0.0.0 --port 8000
```

By changing [generate_sinogram.py](../generate_sinogram.py) different phantoms and geometries can be used. But remember to rebuild the docker container if you've modified the script outside of the docker container, otherwise you'll be running the old script and wondering why nothing works as expected.

# Limitations

There are a few known limitations to the data format and software in general:
- Currently only supports odl `ConeBeamGeometry` geometry.
- Only tested with a curved sensor.
- Only support square sensor pixels
- Currently no translation and no axial shift
- Sample ID scheme is hardcoded.
- `shift.npy` only supports axial shifts