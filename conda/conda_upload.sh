PCG_NAME=face_alignment
USER=1adrianb

mkdir ~/conda-build
conda config --set anaconda_upload no
conda build conda/
anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USER /home/travis/miniconda/envs/test-environment/conda-bld/noarch/face_alignment-1.3.3-py_1.tar.bz2 --force