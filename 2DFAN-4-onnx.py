import torch
import onnx
import onnxruntime
import numpy as np

cuda = True
device = torch.device("cuda:0" if cuda else "cpu")
s = '/home/ubuntu/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip'
onnx_out = "2DFAN4.onnx"

model = torch.jit.load(s,map_location=device)
model.eval()

onnx_model = onnx.load(onnx_out)
onnx.checker.check_model(onnx_model)


#
# dummy_input = torch.randn(1,3,256,256).to(device)
# # dummy_output = model(dummy_input)
#
#
# torch.onnx.export(model, dummy_input, onnx_out, verbose=False, opset_version=11,
#                   training=torch.onnx.TrainingMode.EVAL,do_constant_folding=True,
#                   input_names=['images'],output_names=['outputs'],
#                   dynamic_axes=None
#                   )





# run infer
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
session = onnxruntime.InferenceSession(onnx_out, providers=providers)
dummy_input = torch.randn(1,3,256,256).to(device)
numpy_input = dummy_input.detach().cpu().numpy()
dummy_output = model(dummy_input)
numpy_output = session.run([session.get_outputs()[0].name],{session.get_inputs()[0].name: numpy_input})[0]

print(dummy_output.shape)
print(numpy_output.shape)
# print()
np.testing.assert_allclose(dummy_output.detach().cpu().numpy(),numpy_output,rtol=1e-03, atol=1e-05)





