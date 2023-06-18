# Requirements

We could ensure that the code is available in such environment.
```shell
Linux
RTX 3090 with driver version 470.182.03
gcc 9.4.0
g++ 9.4.0
CUDA 11.1
cuDNN 8.0.4
```


You can run the following command to install the environment
```shell
conda create -n your_env_name python=3.7.0
source activate your_env_name
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorboard==2.8.0
pip install setuptools==52.0.0
pip install protobuf==3.17.2
pip install PyYAML==5.4.1
pip install h5py==3.1.0
pip install pandas==1.1.5
pip install joblib==1.1.0
pip install scikit-learn==1.0.2
pip install statsmodels==0.13.5
```


# Compilation

Part of NMS is implemented in C++. The code can be compiled by

```shell
cd ./libs/utils
python setup.py install --user
cd ../..
```

The code should be recompiled every time you update PyTorch.
