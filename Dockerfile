FROM mambaorg/micromamba:0.26.0
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
ENV PATH="${PATH}:/opt/conda/bin"
WORKDIR /seg_prep
COPY . .
