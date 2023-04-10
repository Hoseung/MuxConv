import numpy as np
import hemul
hemul.USE_FPGA=False
from hemul import heaan
from muxcnn.resnet_HEAAN import ResNetHEAAN
from muxcnn.utils import get_channel_last

from muxcnn.models.ResNet20 import ResNet, BasicBlock
from muxcnn.utils import load_params, load_img, decrypt_result
import torch


# CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

model = ResNet(BasicBlock,[1,1,1])
model.eval()
load_params(model, fn_param="./ResNet8.pt",device='cpu')


logp = 40
logq = 800
logn = 15

# Rotation 미리 준비 
rot_l = [2**i for i in range(15)]
rot_l = rot_l + [2**15-1, 
                 2**15-33, 2**15-32, 2**15-31,
                 2**15-17, 2**15-16, 2**15-15, 
                 2**15-9,2**15-8, 2**15-7] + [3,5,7,9,15,17, 31, 33]

hec = heaan.HEAANContext(logn, logp, logq, load_keys=True, rot_l=rot_l)
fhemodel = ResNetHEAAN(model, hec)


for img in valid_loader:

    ctxt = fhemodel.pack_img_ctxt(img_tensor)
    result = fhemodel(ctxt)
    class_num = decrypt_result(hec, result)
    print("[FHE_CNN] Inference result:", class_num)
    print(f"[FHE_CNN] It's a {classes[class_num]}")

    torch_result = model(img_tensor)
    torch_class = torch.argmax(torch_result)
    print(f"[PyTorch] It's a {classes[torch_class]}")
