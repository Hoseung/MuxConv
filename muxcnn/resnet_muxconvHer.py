import torch.nn as nn
from muxcnn.models import ResNetHer
#from muxcnn.comparator import ApprRelu
from muxcnn.utils import *
from muxcnn.hecnn_par import *
#import math


# class MuxHermitePN():
#     def __init__(self, num_features, eps=1e-5, momentum=0.1):
#         self.num_features = num_features
#         self.eps = eps
                
#     def _normalize(self, ctxt, bn_layer):
#         return parMuxBN_separate(ctxt, bn_layer, outs, nslots)
    
#     def _normalize2(self, ctxt):
#         parMuxBN_separate(ctxt, bn_layer, outs, nslots)
#         return x_hat    

#     def forward(self, ctxt):
#         # Normalized Hermite polynomials
#         h0 = 1
#         h1 = ctxt
#         h2 = (ctxt*ctxt - 1)/math.sqrt(2)

#         # Coefficients of Hermite approximation of ReLU
#         f0 = 1/math.sqrt(2*math.pi)
#         f1 = 1/2
#         f2 = 1/math.sqrt(2*math.pi*2) 
        
#         # Normalize and scale
#         h0_bn = h0 * f0
#         h1_bn = self._normalize(f1*h1)
#         h2_bn = self._normalize2(f2*h2)

#         result = h0_bn + h1_bn + h2_bn
#         return result


class ResNet_MuxConvHer():
    """
    ORG: conv1, bn1, layer1, layer2, layer3, linear 
    
    THIS: conv1, hermitepn, layer1, layer2, layer3, linear
        
        e.g) layer3.0.herPN1.runing_mean2 
    
    """
    def __init__(self, model):
        self.torch_model = model
        self.torch_model.eval()
        self.nslots = 2**15
        
    def forward(self, img_tensor):
        model = self.torch_model
        ctxt, outs0 = self.forward_early_Her(img_tensor)
        
        # Basic blocks
        ctxt, outs1 = self.forward_bb_Her(model.layer1[0], ctxt, outs0)
        ctxt, outs2 = self.forward_bb_Her(model.layer2[0], ctxt, outs1)
        ctxt, outs3 = self.forward_bb_Her(model.layer3[0], ctxt, outs2)
        ctxt = AVGPool(ctxt, outs3, self.nslots) # Gloval pooling
        return self.forward_linear(ctxt, model.linear)

    def forward_early_Her(self, img_tensor):
        model = self.torch_model
        imgl = get_channel_last(img_tensor[0].detach().numpy())
        ki = 1 # initial ki
        hi, wi, ch = imgl.shape

        # early conv and bn
        _, ins0, outs0 = get_conv_params(model.conv1, {'k':ki, 'h':hi, 'w':wi})
        ct_a = MultParPack(imgl, ins0)
        # ctxt, un1 = forward_convbn_par(model.conv1, 
        #                                model.bn1, ct_a, ins0)
        ctxt, un1 = forward_conv_par(model.conv1, ct_a, ins0)
        
        ctxt = hermitePN(ctxt, model.hermitepn, outs0, self.nslots)
        return ctxt, outs0 

    def forward_bb_Her(self, bb:ResNetHer.BasicBlockHer, ctxt_in, outs_in):
        _, ins, outs = get_conv_params(bb.conv1, outs_in)
        ctxt, un = forward_conv_par(bb.conv1, ctxt_in, ins)
        ctxt = hermitePN(ctxt, bb.herPN1, outs, self.nslots)
        #print("1", ctxt[::64])

        _, ins, outs = get_conv_params(bb.conv2, outs)
        ctxt, un = forward_conv_par(bb.conv2, ctxt, ins)
        #ctxt = hermitePN(ctxt, bb.herPN2, outs, self.nslots)
        #print("2", ctxt[:4:64])
        
        # Shortcut
        if len(bb.shortcut) > 0:
            convl = bb.shortcut[0] #Sequential(nn.Conv2d)
            _, ins_, _ = get_conv_params(convl, outs_in)
            shortcut, _ = forward_conv_par(convl, ctxt_in, ins_, 
                                            convl.kernel_size)
        elif len(bb.shortcut) == 0:
            shortcut = ctxt_in

        # Add shortcut
        ctxt += shortcut
        ctxt = hermitePN(ctxt, bb.herPN2, outs, self.nslots)
        # Activation
        #ctxt = self.activation(ctxt)

        return ctxt, outs

    def forward_linear(self, ctxt, linearl:nn.modules.Linear):
        no, ni = linearl.weight.shape

        weight_vec = np.zeros(self.nslots)
        weight_vec[:no*ni] = np.ravel(linearl.weight.detach().numpy())

        # make 10 copies of 64 flattened values
        for i in range(ceil(np.log2(no))):
            ctxt += np.roll(ctxt, 2**i*ni)

        # multiply 64 * 10 at once
        # AVGPool에서 S_vec를 조절하면 이 단계의 일부를 미리 수행할 수 있음 !!
        ctxt = weight_vec * ctxt

        # Sum 64 numbers each 
        for j in range(int(np.log2(ni))):
            ctxt += np.roll(ctxt, -2**j)

        return ctxt

    def __call__(self, img_tensor):
        return self.forward(img_tensor)