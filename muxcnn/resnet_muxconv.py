import torch.nn as nn
from muxcnn.models import ResNet20
from hemul.comparator import ApprRelu
from muxcnn.utils import *
from muxcnn.hecnn_par import *
from copy import copy
class ResNet_MuxConv():
    def __init__(self, model, alpha=12, debug=False):
        self.torch_model = model
        self.torch_model.eval()
        self.nslots = 2**15
        self.debug = debug
        
        self._set_activation(alpha=alpha, xmin=-40, xmax=40, min_depth=True, debug=True)
        
    def _set_activation(self, *args, **kwargs):
        self.activation = ApprRelu(*args, **kwargs)

    def forward(self, img_tensor):
        model = self.torch_model
        ctxt, outs0 = self.forward_early(img_tensor)
        if self.debug:
            self._debug_log.append(["forward_early", copy(ctxt), outs0])
        
        # Basic blocks
        ctxt, outs1 = self.forward_bb(model.layer1[0], ctxt, outs0)
        if self.debug:
            self._debug_log.append(["basic0", copy(ctxt), outs1])
        ctxt, outs2 = self.forward_bb(model.layer2[0], ctxt, outs1)
        if self.debug:
            self._debug_log.append(["basic1", copy(ctxt), outs2])
        ctxt, outs3 = self.forward_bb(model.layer3[0], ctxt, outs2)
        if self.debug:
            self._debug_log.append(["basic2", copy(ctxt), outs3])
        ctxt = AVGPool(ctxt, outs3, self.nslots) # Gloval pooling
        if self.debug:
            self._debug_log.append(["AVGPool", copy(ctxt)])
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
        # AVGPool에서 S_vec를 조절하면 이 단계의 일부를 미리 수행할 수 있음 !!
        ctxt = weight_vec * ctxt

        # Sum 64 numbers each 
        for j in range(int(np.log2(ni))):
            ctxt += np.roll(ctxt, -2**j)

        return ctxt

    def __call__(self, img_tensor, debug=False):
        self.debug = debug
        if self.debug:
            self._debug_log = []
        return self.forward(img_tensor)