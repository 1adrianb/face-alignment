from net_s3fd import s3fd
from of_net_s3fd import ofs3fd
import numpy as np
import torch
import oneflow
import globalVar as gl
gl._init()


torch_model_path = '/home/ubuntu/work/oneflow/face-alignment/torch_model_hub/s3fd-619a316812.pth'
oneflow_model_path = '/home/ubuntu/work/oneflow/face-alignment/oneflow_model_hub/s3fd-619a316812'
torch_model = s3fd()
oneflow_model = ofs3fd()


torch_parameters = torch.load(torch_model_path)
oneflow_parameters = torch_parameters.copy()
torch_model.load_state_dict(torch_parameters)
for key, value in torch_parameters.items():
    val = value.detach().cpu().numpy()
    oneflow_parameters[key] = val
    # print("key:", key, "value.shape", val.shape)
oneflow_model.load_state_dict(oneflow_parameters)
oneflow.save(oneflow_model.state_dict(), oneflow_model_path)


def tensor2numpy(torchtensor,oftensor):
    return torchtensor.detach().cpu().numpy(),oftensor.detach().cpu().numpy()

torch_model.eval()
oneflow_model.eval()
np.random.seed(42)
for i in range(10):
    inp = np.random.randn(1,3,256,256).astype(np.float32)
    torch_inp = torch.from_numpy(inp)
    oneflow_inp = oneflow.from_numpy(inp)
    # print(torch_inp.dtype,oneflow_inp.dtype)
    # print((torch_inp.detach().cpu().numpy()==oneflow_inp.detach().cpu().numpy()).all())
    # print(torch_inp)
    # print(oneflow_inp)
    torch_out = torch_model(torch_inp)
    oneflow_out = oneflow_model(oneflow_inp)
    allVars = gl.get().copy()
    for index,(i,j) in enumerate(zip(*allVars)):
        if isinstance(i,torch.Tensor) and isinstance(j,oneflow.Tensor):
            print(index)
            np.testing.assert_allclose(*tensor2numpy(i,j),rtol=1e-03, atol=1e-05)

        elif isinstance(i,tuple) and isinstance(j,tuple):
            for ii,jj in zip(i,j):
                print(index)
                np.testing.assert_allclose(*tensor2numpy(ii,jj),rtol=1e-03, atol=1e-05)
