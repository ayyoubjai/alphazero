FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# Python 3.10 + pip
RUN apt-get update && apt-get install -y software-properties-common \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update && apt-get install -y python3.10 python3.10-dev python3.10-distutils curl git \
 && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN python3 -m pip install --upgrade pip

# Install Graphviz system package (if needed at runtime)
RUN apt-get update && apt-get install -y graphviz && rm -rf /var/lib/apt/lists/*

# Copy metadata first (caching)
COPY pyproject.toml poetry.lock* README.md /app/

# Install Poetry
RUN python3 -m pip install poetry

# Install heavy binary wheel(s) first (Torch with CUDA)
RUN python3 -m pip install --no-cache-dir --retries 5 --timeout 60 \
    torch==2.4.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Copy source and examples
COPY src/ src/
COPY examples/ examples/

# Tell Poetry to install into the system env and install dependencies & your package
RUN poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi --without dev
