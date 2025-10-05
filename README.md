# CT Reconstruction & Streaming Server (MITO)

Tools for **loading MITO CT data**, **streaming acquisition geometry + sinogram slices**, and **reconstructing 3D volumes** (Adjoint, FBP, Landweber).  
Built on **ODL** (custom ASTRA curved branch), **ASTRA Toolbox (curved geometry)**, **FastAPI**, and **PyVista** (off-screen).

> Works with **3D Slicer** as a client, but you can also drive it via cURL or Python.

---

![Pipeline Architecture: 3D Slicer with FastAPI Streaming and Reconstruction Server](Pipeline_Architecture.jpg)

## Repository Structure

    .
    ├── dataset.py               # MITO dataset loader (PyTorch Dataset with ODL operators)
    ├── reconstruction.py        # Standalone script to run reconstruction (adjoint, fbp, landweber)
    ├── odl_stream_server.py     # FastAPI server for streaming geometry/sinogram slices and running reconstructions
    ├── slicer_backend_config.json # Configuration file used to populate the menu in 3D Slicer
    ├── Dockerfile               # The dockerfile    
    └── README.md

## Functionality Summary — odl_stream_server.py

-   Starts a **FastAPI server** for CT streaming and reconstruction.
-   **Loads CT sample data** (sinogram, angles, shifts, metadata) from the MITO dataset.
-   **Builds ODL cone-beam geometry** for the scan (source, detector, curvature, pitch).
-   Provides **API endpoints** to:
    -   /stream_window → stream geometry and detector panels for visualization.
    -   /get_sinogram_slice/{index} → return a single projection as NRRD.
    -   /full_geometry and /full_trajectory.json → serve precomputed geometry and source paths.
    -   /select_sample → switch dataset (tree/disk).
    -   /run_reconstruction → execute reconstruction.py with chosen method (adjoint, FBP, Landweber).

-   **Saves outputs** (VTP, JSON, NRRD) to output/ and images/ directories.
-   Used by **3D Slicer** as a backend to visualize and reconstruct CT data dynamically.

## Functionality Summary — reconstruction.py

-   Reconstructs a **3D CT volume** from the **MITO dataset** for a specified tree and disk.

-   Supports **three reconstruction methods**:

    -   adjoint → simple backprojection (Aᵗb)

    -   fbp → filtered backprojection using ODL's fbp_op

    -   landweber → iterative gradient-based reconstruction (x ← x + ωAᵗ(b-Ax))

-   Accepts parameters such as number of iterations (niter) and relaxation factor (omega) via JSON.

-   Provides **verbose progress logging**, including iteration percentage for Landweber.

-   Optionally writes **progress updates to a JSONL file** (--progress_path).

-   Loads geometry and sinogram data from the **MITO dataset** via the dataset.MITO class.

-   Saves the reconstructed 3D volume as an **NRRD file** with correct voxel spacing metadata.

-   Designed for **integration with the FastAPI server** (odl_stream_server.py) to enable remote or containerized reconstruction.

## MITO Dataset Layout

Expected per-sample directory (bind-mounted read-only into the container):

    ├─ sinogram.npy            
    ├─ angles.npy              
    ├─ axial_positions.npy     
    ├─ shifts.npy              
    └─ metadata.json           

## Docker (CUDA 11.3 + cuDNN8)

This repo includes a Dockerfile that installs:

- Miniforge (conda-forge) with pinned numpy/pandas/matplotlib

- ASTRA Toolbox 2.1.3 (curved label)

- PyTorch 1.12.1+cu113

- VTK 9.2.6, FastAPI, PyVista, SciPy, pynrrd

- ODL custom branch: astra_cylcone_binding

## Clone the repository

    git clone https://github.com/ouaburst/CT_GUI_Docker.git

## Build


Inside the directory `CT_GUI_Docker/` type:

    docker build -t woodscan:cuda113 .

 ## Run the API server (GPU)   

    docker run --rm -it -p 8000:8000 \
     --gpus all \
     -v /media/Store-SSD:/media/Store-SSD:ro \
     woodscan:cuda113 \
     python -m uvicorn odl_stream_server:app --host 0.0.0.0 --port 8000

Notes:

-   **--rm**: Remove container after exit (keeps system clean).

-   **-it**: Interactive terminal (view logs in real time).

-   **-p 8000:8000**: Map container port 8000 → host port 8000 (<http://localhost:8000>).

-   **--gpus all**: Use all available GPUs (requires nvidia-container-toolkit).

    -   Limit to one GPU: --gpus '"device=0"' or -e CUDA_VISIBLE_DEVICES=0.

-   **-v /media/Store-SSD:/media/Store-SSD:ro**: Mount dataset (read-only).

-   **woodscan:cuda113**: Docker image (built with CUDA 11.3, ODL, ASTRA, FastAPI).

-   **python -m uvicorn odl_stream_server:app**: Runs the FastAPI server inside the container.

## API Endpoints ##

Root:

-   **GET /** --- status + usage hints

Geometry / Sinogram:

-   **GET /stream_window?i=100&n=10** --- sources, detector panels, FOV rays, Bézier surfaces

-   **GET /full_geometry** --- compact precomputed geometry JSON

-   **GET /full_trajectory.json** --- source positions for all views

-   **GET /get_sinogram_slice/{index}** --- one projection as .nrrd

## License ##

MIT --- see \<LICENSE>.

## Acknowledgments ##

-   **ODL** team for operator framework

-   **ASTRA Toolbox** developers for GPU projectors

-   **PyVista** team for VTK-based visualization tools