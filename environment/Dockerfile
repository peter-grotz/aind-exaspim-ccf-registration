# hash:sha256:3bd65a6ed0c112fdaa291b29e439fdbfa6a3a2157738a0a0e532d5377a2ce126
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

RUN conda install -y --channel=conda-forge \
        ants==2.6.5 \
        s3fs==2023.9.2 \
        pyopenssl>=23 \
    && conda clean -ya \
    && which antsApplyTransforms

RUN pip install -U --no-cache-dir \
    antspyx==0.5.4 \
    ome-zarr==0.8.2 \
    simpleitk==2.3.0 \
    zarr==2.16.1 \
    argschema==3.0.4 \
    openpyxl==3.1.5 \
    aind-data-schema==1.3.0 \
    scikit-image==0.19.3 \
    tifffile==2022.10.10 \
    bokeh==2.4.2 \
    xarray_multiscale==1.1.0 \
    dask[distributed]==2022.11.1 \
    cloud-volume[all_codecs]==11.0.2 \
    matplotlib==3.7.3 \
    natsort==8.4.0 \
    crackle-codec==0.26.0 \
    aicsimageio@git+https://github.com/camilolaiton/aicsimageio.git@feature/zarrwriter-multiscales-daskjobs
    
