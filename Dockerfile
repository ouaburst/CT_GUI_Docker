# ==============================
# CUDA 11.3 + cuDNN8 base
# ==============================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04

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
      python=3.12 \
      numpy=2.4.3 \
      pandas=2.3.3 \
      matplotlib-base=3.10.9 \
  #&& mamba install -y -c wjpalenstijn/label/curved astra-toolbox=2.4.1 \
  && mamba install -y -c astra-toolbox -c nvidia astra-toolbox==2.4.1 \
  && conda clean -a -y

# ------------------------------
# Pip packages
#   - VTK cp310 wheels
#   - Torch 1.12.1 + cu113 wheels
#   - FastAPI stack, SciPy, pynrrd, PyVista
# ------------------------------
RUN pip install --no-cache-dir \
      vtk==9.6.1 \
      #torch==2.1.0 \
      #torchvision==0.16.0 \
      #torchaudio==2.1.0 \
      #--extra-index-url https://download.pytorch.org/whl/cu121 \
      pillow==12.1.1 \
  && pip install --no-cache-dir \
      fastapi uvicorn[standard] starlette requests scipy pynrrd pyvista

RUN pip install --no-cache-dir \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ------------------------------
# ODL (custom ASTRA curved branch)
# ------------------------------
#RUN pip install --no-cache-dir odl==1.0.0
#RUN git clone -b astra_cylcone_binding https://github.com/wjp/odl.git /opt/odl \
# && pip install -e /opt/odl

RUN git clone -b astra-curved-detector https://github.com/NogginBops/odl.git /opt/odl \
 && pip install -e /opt/odl

# ------------------------------
# Workspace
# ------------------------------
WORKDIR /workspace
COPY . /workspace

# Expose server port
EXPOSE 8000

# Default command (we override at run)
CMD ["python3"]