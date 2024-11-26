import torch
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

model = torchvision.models.detection.maskrcnn_resnet50_fpn(
    weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT
)
model.eval()

input_tensor = torch.randn(1, 3, 256, 256)

onnx_file_path = f"model.onnx"

torch.onnx.export(model.cpu(),
                  input_tensor.cpu(),
                  onnx_file_path,
                  export_params=True,
                  do_constant_folding=False,
                  input_names = ['input'],
                  output_names = ['boxes', 'labels', 'scores', 'masks'],
                  dynamic_axes={'input': {2 : 'height', 3 : 'width'}}
                 )