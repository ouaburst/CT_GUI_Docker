# ==============================
# CUDA 11.3 + cuDNN8 base
# ==============================
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda
# PyVista headless rendering
ENV PYVISTA_OFF_SCREEN=1
# Avoid matplotlib pulling Qt
ENV MPLBACKEND=Agg

# ------------------------------
# System deps
# ------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ------------------------------
# Miniforge (conda-forge first)
# ------------------------------
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
 && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
 && rm -f /tmp/miniforge.sh \
 && conda config --system --set channel_priority flexible \
 && conda config --system --add channels conda-forge \
 && conda install -y mamba -n base -c conda-forge

# ------------------------------
# Conda core (pin versions)
# ------------------------------
RUN mamba install -y \
      python=3.10 \
      numpy=1.26.3 \
      pandas=2.2.2 \
      matplotlib-base=3.8.0 \
  && mamba install -y -c wjpalenstijn/label/curved astra-toolbox=2.1.3 \
  && conda clean -a -y

# ------------------------------
# Pip packages
#   - VTK cp310 wheels
#   - Torch 1.12.1 + cu113 wheels
#   - FastAPI stack, SciPy, pynrrd, PyVista
# ------------------------------
RUN pip install --no-cache-dir \
      vtk==9.2.6 \
      torch==1.12.1+cu113 \
      torchvision==0.13.1+cu113 \
      torchaudio==0.12.1 \
      --extra-index-url https://download.pytorch.org/whl/cu113 \
  && pip install --no-cache-dir \
      fastapi uvicorn[standard] starlette requests scipy pynrrd pyvista

# ------------------------------
# ODL (custom ASTRA curved branch)
# ------------------------------
RUN git clone -b astra_cylcone_binding https://github.com/wjp/odl.git /opt/odl \
 && pip install -e /opt/odl

# ------------------------------
# Workspace
# ------------------------------
WORKDIR /workspace
COPY . /workspace

# Make project-imports work (e.g. 'import odl_utils')
ENV PYTHONPATH=/workspace:${PYTHONPATH}

# Expose server port
EXPOSE 8000

# Default command (we override at run)
CMD ["python3"]