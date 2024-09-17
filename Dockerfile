FROM mambaorg/micromamba:latest
USER root
COPY environment.yml .
RUN apt-get update -qq && apt-get install -y \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    procps
ARG MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH

RUN micromamba env create -f environment.yml \
    && micromamba clean --all --yes
WORKDIR /seg_prep
COPY . .
