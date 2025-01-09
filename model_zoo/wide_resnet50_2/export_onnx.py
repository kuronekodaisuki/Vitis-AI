import os
import re
import sys
import argparse
import time
import pdb
import random
#from pytorch_nndct.apis import torch_quantizer
import torch
import torchvision
from torchvision import models, transforms 
#from torchvision.models import wide_resnet50_2

model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weigths.IMAGENET1K_V2)
input = torch.randn(1,, 1,, 32, 32)
onnx_model = torch.onnx.dynamo_export(model, input)
onnx_model.save("wide_resnet50_2-IMAGENET1K_V2.onnx")