import numpy as np
import torch
from torch.nn import functional as F
from hlee_model import ConvNeuralNet_simple as cnn
from hlee_utils import *

class TorchConv_infer(cnn):
    def __init__(self, num_classes=10, activation=F.relu, fn_param="", device='cpu'):
        """
        FHE랑 비교용도의 Pytorch model
        ho: height out
        wo: width out
        co: channel out
        """
        super().__init__(num_classes, activation=F.relu)
        self.fn_param = fn_param
        self._set_params(device)
        self.eval()

    def layer_info(self, name):
        layer = getattr(self, name)
        return get_channel_last(np.array(layer.weight.detach())), layer.weight.shape
        
        U = self.conv_layer1.weight
        Ut = self.conv_layer1.weight.clone().detach()
        self.U_multconv = get_channel_last(U)
        self.U_torchconv = Ut.type(torch.DoubleTensor)
        print(f"[cnn] U_multconv  : {self.U_multconv.shape}")
        print(f"[cnn] U_torchconv : {self.U_torchconv.shape}")

    def _set_params(self, device):
        trained_param = torch.load(self.fn_param, map_location = torch.device(device))
        trained_param = {key : value.cpu()   for key,value in trained_param.items()}
        params_np     = {key : value.numpy() for key,value in trained_param.items()}
        self.load_state_dict(trained_param)

    def load_img(self, fname, hi=None, wi=None):
        image = cv2.imread(fname)
        if hi is not None and wi is not None:
            image = cv2.resize(image,(hi,wi))
        img = get_channel_first(image)
        img = torch.tensor(img)
        self.img = img.type(torch.FloatTensor)
        return self.img

    def __call__(self, input, name=None):
        """
        forward specified layer
        """
        if name is None:
            return cnn.__call__(input)
        else:
            layer = getattr(self, name)
            return layer(input)

    def get_dims(self):
        pass
    
    #def TorchConv(self, fname):
    #    
    #    return F.conv2d(img,self.U_torchconv,padding="same")

    # def TorchConv_single(self,img):
    #     #image = cv2.imread(fname)
    #     #image = cv2.resize(image,(self.ho,self.wo))
    #     img = get_channel_first(img)
    #     img = torch.tensor(img)
    #     img = img.type(torch.DoubleTensor)
    #     return F.conv2d(img,self.U_torchconv,padding="same")
        

