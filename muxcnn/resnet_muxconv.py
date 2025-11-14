"""
ResNet implementation using multiplexed convolution for FHE.

This module provides a ResNet architecture adapted for homomorphic
encryption using multiplexed convolution operations.
"""

import torch.nn as nn
from muxcnn.models import ResNet20
from muxcnn.comparator import ApprRelu
from muxcnn.utils import *
from muxcnn.hecnn_par import *
from muxcnn.hecnn import DEFAULT_NSLOTS

class ResNet_MuxConv():
    def __init__(self, model, alpha=12):
        self.torch_model = model
        self.torch_model.eval()
        self.nslots = DEFAULT_NSLOTS
        
        self._set_activation(alpha=alpha, xmin=-10, xmax=10, min_depth=True)
        
    def _set_activation(self, *args, **kwargs):
        self.activation = ApprRelu(*args, **kwargs)

    def forward(self, img_tensor):
        model = self.torch_model
        ctxt, outs0 = self.forward_early(img_tensor)
        
        # Basic blocks
        ctxt, outs1 = self.forward_bb(model.layer1[0], ctxt, outs0)
        ctxt, outs2 = self.forward_bb(model.layer2[0], ctxt, outs1)
        ctxt, outs3 = self.forward_bb(model.layer3[0], ctxt, outs2)
        ctxt = AVGPool(ctxt, outs3, self.nslots) # Global pooling
        return self.forward_linear(ctxt, model.linear)

    def forward_early(self, img_tensor):
        model = self.torch_model
        imgl = get_channel_last(img_tensor[0].detach().numpy())
        ki = 1 # initial ki
        hi, wi, ch = imgl.shape

        # early conv and bn
        _, ins0, outs0 = get_conv_params(model.conv1, {'k':ki, 'h':hi, 'w':wi})
        ct_a = MultParPack(imgl, ins0)
        ctxt, un1 = forward_convbn_par(model.conv1, 
                                       model.bn1, ct_a, ins0)
        ctxt = self.activation(ctxt)
        return ctxt, outs0 

    def forward_bb(self, bb:ResNet20.BasicBlock, ctxt_in, outs_in):
        _, ins, outs = get_conv_params(bb.conv1, outs_in)
        ctxt, un = forward_convbn_par(bb.conv1,
                                      bb.bn1, ctxt_in, ins)
        ctxt = self.activation(ctxt)

        _, ins, outs = get_conv_params(bb.conv2, outs)
        ctxt, un = forward_convbn_par(bb.conv2,
                                      bb.bn2, ctxt, ins)
        # Shortcut
        if len(bb.shortcut) > 0:
            convl, bnl = bb.shortcut
            _, ins_, _ = get_conv_params(convl, outs_in)
            shortcut, _ = forward_convbn_par(convl, bnl, ctxt_in, ins_, 
                                            convl.kernel_size)
        elif len(bb.shortcut) == 0:
            shortcut = ctxt_in

        # Add shortcut
        ctxt += shortcut
        # Activation
        ctxt = self.activation(ctxt)

        return ctxt, outs

    def forward_linear(self, ctxt, linearl:nn.modules.Linear):
        no, ni = linearl.weight.shape

        weight_vec = np.zeros(self.nslots)
        weight_vec[:no*ni] = np.ravel(linearl.weight.detach().numpy())

        # make 10 copies of 64 flattened values
        for i in range(ceil(np.log2(no))):
            ctxt += np.roll(ctxt, 2**i*ni)

        # multiply 64 * 10 at once
        # Note: Part of this step can be performed in advance by adjusting S_vec in AVGPool
        ctxt = weight_vec * ctxt

        # Sum 64 numbers each 
        for j in range(int(np.log2(ni))):
            ctxt += np.roll(ctxt, -2**j)

        return ctxt

    def __call__(self, img_tensor):
        return self.forward(img_tensor)