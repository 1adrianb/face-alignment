import numpy as np
import oneflow
import torch


def tensor2numpy(torchtensor,oftensor):
    return torchtensor.detach().cpu().numpy(),oftensor.detach().cpu().numpy()

torch_model_path = '/home/ubuntu/work/oneflow/face-alignment/torch_model_hub/s3fd-619a316812.pth'
oneflow_model_path = '/home/ubuntu/work/oneflow/face-alignment/oneflow_model_hub/s3fd-619a316812'

torch_model = torch.load(torch_model_path)
oneflow_model = oneflow.load(oneflow_model_path)

assert torch_model.keys()==oneflow_model.keys()

for k in torch_model.keys():
    tp = torch_model[k]
    op = oneflow_model[k]
    # print(tp,op)
    np.testing.assert_allclose(*tensor2numpy(tp,op))

print("PASS")