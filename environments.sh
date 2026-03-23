export CUDA_HOME=/usr/local/cuda

export CUDNN_INCLUDE_DIR=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/include
export CUDNN_LIB_DIR=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib

export NCCL_INCLUDE_DIR=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/nccl/include
export NCCL_LIB_DIR=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/nccl/lib

export CPATH=$NCCL_INCLUDE_DIR:$CUDNN_INCLUDE_DIR:$CUDA_HOME/include
export LIBRARY_PATH=$NCCL_LIB_DIR:$CUDNN_LIB_DIR:$CUDA_HOME/lib64
export LD_LIBRARY_PATH=$NCCL_LIB_DIR:$CUDNN_LIB_DIR:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

conda install cmake ninja
conda install -c nvidia cuda-nvcc cuda-toolkit
# depending on your cuda version
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
# misc
pip install megatron-core hydra-core loguru attrs fvcore nvidia-ml-py imageio[ffmpeg] pandas wandb psutil ftfy regex transformers webdataset
# transformer_engine
pip install --no-build-isolation transformer_engine[pytorch]

cd flash-attention
git checkout v2.7.4.post1
MAX_JOBS=4 python setup.py install
