# import torch
# import oneflow as flow
#
# a1 = torch.randn(3,3)
# a2 = torch.randn(3)
#
# print(torch.matmul(a1,a2))
#
# b1 = flow.randn(3,3)
# b2 = flow.randn(3)
#
# print(a2.shape,b2.shape)
# print(flow.matmul(b1,b2))


import torch
import oneflow as flow
import numpy as np

a = torch.randn(2,)
b = torch.randn(2,)
c = np.array([b[1]-a[1],b[0]-a[0],a.shape[0]],dtype=np.int32)
print(c)

a = flow.randn(2,)
b = flow.randn(2,)
c = np.array([b[1]-a[1],b[0]-a[0],a.shape[0]],dtype=np.int32)
print(c)
