# TTN-VQC (Tensor-Train Network + Variational Quantum Circuit)

The package provides an implementation of TTN-VQC to corroborate our theoretical work. 

```
git clone https://github.com/uwjunqi/TTN-VQC.git
cd TTN-VQC
```

## Installation

The main dependencies include *pytorch*, *pennylane*. To download and install *tc*:

```
git clone https://github.com/uwjunqi/Pytorch-Tensor-Train-Network.git
cd Pytorch-Tensor-Train-Network
python setup.py install
```

## Usage of Codes

reg_add_noise.py: generate noisy image data

reg_ttn_vqc.py: the implementation of TTN-VQC model

reg_pca_vqc.py: the implementation of PCA-VQC model

## Reference:

This package is related to our recently submitted paper 

Jun Qi, Chao-Han Huck Yang, Pin-Yu Chen, Min-Hsiu Hsieh, "Theoretical Error Performance Analysis for Variational Quantum Circuit Based Functional Regression," npj Quantum Information, Nature Publishing Group UK London,  Vol. 9, no. 4, 2023
