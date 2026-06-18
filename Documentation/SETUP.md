# How to generate ConeBeamGeometry data

Only support square sensor pixels?

sinogram.npy - The sinogram data. Shape: ()
angles.npy - The source/detector angles. Shape: ()
axial_position.npy - The z axis position of the source/detector. Shape: ()
shifts.npy - Fine scale shifts of the source/detector position. Shape: ()
metadata.json - Describes the geometry setup.

Run `docker run --rm -it -p 8000:8000  --gpus all  -v /media/Store-SSD:/media/Store-SSD:ro -v /home/julius/CT_GUI_Docker/test:/test  woodscan:cuda121  python generate_sinogram.py`
