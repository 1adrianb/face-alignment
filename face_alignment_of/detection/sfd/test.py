from net_s3fd import s3fd
from of_net_s3fd import ofs3fd
import numpy as np
import torch
import oneflow

torch_model_path = '/home/ubuntu/work/oneflow/face-alignment/torch_model_hub/s3fd-619a316812.pth'
oneflow_model_path = '/home/ubuntu/work/oneflow/face-alignment/oneflow_model_hub/s3fd-619a316812'
torch_model = s3fd()
oneflow_model = ofs3fd()


torch_parameters = torch.load(torch_model_path)
torch_model.load_state_dict(torch_parameters)
for key, value in torch_parameters.items():
    val = value.detach().cpu().numpy()
    torch_parameters[key] = val
    # print("key:", key, "value.shape", val.shape)
oneflow_model.load_state_dict(torch_parameters)
oneflow.save(oneflow_model.state_dict(), oneflow_model_path)




np.random.seed(42)
for i in range(10):
    inp = np.random.randn(1,3,256,256).astype(np.float32)
    torch_inp = torch.from_numpy(inp)
    oneflow_inp = oneflow.from_numpy(inp)
    torch_out = torch_model(torch_inp)
    oneflow_out = oneflow_model(oneflow_inp)
    assert len(torch_out)==len(oneflow_out)
    length = len(torch_out)
    for j in range(length):
        to,oo = torch_out[j],oneflow_out[j]
        print(to.shape)
        print(oo.shape)
        np.testing.assert_allclose(to.detach().cpu().numpy(),oo.detach().cpu().numpy())