import oneflow
import torch

import torch.nn as tnn
import oneflow.nn as onn

torch_model_path = '/home/ubuntu/work/oneflow/face-alignment/torch_model_hub/s3fd-619a316812.pth'
torch_parameters = torch.load(torch_model_path)
for key, value in torch_parameters.items():
    print(value.shape)
    print(key)
weight = torch_parameters['conv1_1.weight']
bias = torch_parameters['conv1_1.bias']
conv1_1 = tnn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
conv2_1 = onn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

