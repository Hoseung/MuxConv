import torch.nn as nn
from muxcnn.models import ResNet20
from muxcnn.comparator import ApprRelu
from muxcnn.utils import *
from muxcnn.hecnn_par import *
import math


class MuxHermitePN():
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        
        # Batch normalization layer
        #self.batch_norm = nn.BatchNorm2d(num_features)
        self.eps = eps
        self.momentum = momentum
        
        
        ## Parameter는 훈련된 모델에서 가져옴. 아래는 무시. 
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def _normalize(self, x):
        # Use running statistics
        batch_mean = self.running_mean.view(1, self.num_features, 1, 1)
        batch_var = self.running_var.view(1, self.num_features, 1, 1)
        
        # Normalize
        x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        return x_hat    
        
    def _scale(self, x):
        return self.gamma.view(1, self.num_features, 1, 1) * x + self.beta.view(1, self.num_features, 1, 1)


    def forward(self, x):
        # print("HermitePN", x.min(), x.max())
        H0 = 1 
        H1 = x
        H2 = x**2 - 1
        #if n==3: H3 = 8*x**3 - 12*x

        # Normalized Hermite polynomials
        h0 = H0
        h1 = H1
        h2 = H2/math.sqrt(2)
        #if n==3: h3 = H3/math.sqrt(6)

        # h1에 normalize?
        # 아님 f1*h1에 normalize?

        h0_bn = h0
        h1_bn = self._normalize(h1)
        h1 = ctxt 
        
        # gamma, beta
        mu, var = get_bn_params(self.bn1,)
        
        (x - mu)/np.sqrt(var + self.eps)
        
        h2_bn = self._normalize(h2)

        # Coefficients of Hermite approximation of ReLU
        f0 = 1/math.sqrt(2*math.pi)
        f1 = 1/2
        f2 = 1/math.sqrt(2*math.pi*2) 
        
        # scale(gamma)과 bias(beta)는 learnable parameter임. 
        # gamma1, gamma2, gamma3, ... 로 여러 gamma를 유지할 것이 아니므로, 모든 term을 합친 뒤 최종적으로 gamma와 beta 적용.
        result = self._scale(f0*h0_bn + f1*h1_bn + f2*h2_bn) 
        # print("HermitePN result", result.min(), result.max())
        return result


class ResNet_MuxConvHer():
    def __init__(self, model, alpha=12):
        self.torch_model = model
        self.torch_model.eval()
        self.nslots = 2**15
        
        #self._set_activation(alpha=alpha, xmin=-10, xmax=10, min_depth=True)
        self.herPN1 = MuxHermitePN()
        
    # def _set_activation(self, *args, **kwargs):
    #     self.activation = ApprRelu(*args, **kwargs)

    def forward(self, img_tensor):
        model = self.torch_model
        ctxt, outs0 = self.forward_early(img_tensor)
        
        # Basic blocks
        ctxt, outs1 = self.forward_bb(model.layer1[0], ctxt, outs0)
        ctxt, outs2 = self.forward_bb(model.layer2[0], ctxt, outs1)
        ctxt, outs3 = self.forward_bb(model.layer3[0], ctxt, outs2)
        ctxt = AVGPool(ctxt, outs3, self.nslots) # Gloval pooling
        return self.forward_linear(ctxt, model.linear)

    def forward_early(self, img_tensor):
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
        # ctxt = self.activation(ctxt)
        ctxt = self.herPN1(ctxt)
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

    def __call__(self, img_tensor):
        return self.forward(img_tensor)