PCG_NAME=face_alignment
USER=1adrianb

mkdir ~/conda-build
conda config --set anaconda_upload no
export CONDA_BUILD_PATH=~/conda-build
conda build --output-folder $CONDA_BUILD_PATH conda/
anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USER $CONDA_BUILD_PATH/$PKG-NAME-1-tar.bz2 --force