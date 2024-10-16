# Use Python 3.12-slim as the base image
FROM python:3.12-slim

# Set environment variables for Python to handle encoding properly
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for building FAISS and Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    wget \
    libomp-dev \
    libgfortran5 \
    git \
    swig \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install setuptools
RUN pip install --upgrade pip setuptools

# Verify SWIG installation (optional but recommended)
RUN swig -version

# Set the working directory
WORKDIR /app

# Install NumPy via pip to ensure NumPy headers are available
RUN pip install --no-cache-dir numpy

# Install FAISS from source (disable CUDA)
RUN git clone https://github.com/facebookresearch/faiss.git \
    && cd faiss \
    && git checkout v1.7.3 \
    && cmake -B build \
        -DFAISS_ENABLE_PYTHON=ON \
        -DFAISS_ENABLE_C_API=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DFAISS_ENABLE_GPU=OFF \
        -DPython_EXECUTABLE=$(which python) \
    && cmake --build build --target faiss -j $(nproc) \
    && cmake --build build --target swigfaiss -j $(nproc) \
    && cd build/faiss/python \
    && python setup.py install \
    && python -c "import faiss; print(faiss.__version__)" \ 
    && cd /app \
    && rm -rf faiss

# Copy requirements.txt into the container
COPY requirements.txt /app/

# Remove faiss-cpu from requirements.txt if present
RUN sed -i '/faiss-cpu/d' requirements.txt

# Install other Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app/

# Expose port 8501 for Streamlit
EXPOSE 8501

# Set the entrypoint to run the Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
