import numpy as np
from math import ceil
import torch.nn as nn
from typing import Dict
from muxcnn.models import ResNet20
from hemul.comparator_fhe import ApprRelu_FHE
from hemul.ciphertext import CiphertextStat
from hemul.utils import key_hash
from muxcnn.utils import get_q, get_conv_params, get_channel_last
from muxcnn.hecnn_par import (MultParPack, 
                                parMuxBN, 
                                tensor_multiplexed_selecting, 
                                Vec, 
                                ParMultWgt)
from muxcnn.hecnn_par import AVGPool



class ResNetFHE():
    def __init__(self, model, alpha=12):
        self.torch_model = model
        self.torch_model.eval()
        self.nslots = 2**15
        self.alpha=alpha
    
    def set_agents(self, context, ev, encoder, encryptor):
        self.context = context
        self.ev = ev
        self.encoder = encoder
        self.encryptor = encryptor
        self._set_activation(alpha=self.alpha, xmin=-10, xmax=10, min_depth=True)
        
    def _set_activation(self, *args, **kwargs):
        self.activation = ApprRelu_FHE(self.ev, *args, **kwargs)

    def forward(self, img_tensor):
        model = self.torch_model
        ctxt, outs0 = self.forward_early(img_tensor)
        
        # Basic blocks
        ctxt, outs1 = self.forward_bb(model.layer1[0], ctxt, outs0)
        ctxt, outs2 = self.forward_bb(model.layer2[0], ctxt, outs1)
        ctxt, outs3 = self.forward_bb(model.layer3[0], ctxt, outs2)
        ctxt = AVGPool(ctxt, outs3, self.nslots) # Gloval pooling
        return self.forward_linear(ctxt, model.linear)

    def pack_img_ctxt(self, img_tensor):
        """ For convenience
        """
        model = self.torch_model
        imgl = get_channel_last(img_tensor[0].detach().numpy())
        ki = 1 # initial ki
        hi, wi, ch = imgl.shape

        # early conv and bn
        _, ins0, outs0 = get_conv_params(model.conv1, {'k':ki, 'h':hi, 'w':wi})
        ct_a = MultParPack(imgl, ins0)
        return self.encryptor.encrypt(ct_a)


    def forward_early(self, ct_a, ki, hi, wi):
        model = self.torch_model
        _, ins0, outs0 = get_conv_params(model.conv1, {'k':ki, 'h':hi, 'w':wi})
        ctxt, un1 = self.forward_convbn_par_fhe(model.conv1, 
                                        model.bn1, ct_a, ins0)
        ctxt = self.activation(ctxt)
        return ctxt, outs0 

    def forward_bb(self, bb:ResNet20.BasicBlock, ctxt_in, outs_in):
        _, ins, outs = get_conv_params(bb.conv1, outs_in)
        ctxt, un = self.forward_convbn_par_fhe(bb.conv1,
                                        bb.bn1, ctxt_in, ins)
        ctxt = self.activation(ctxt)

        _, ins, outs = get_conv_params(bb.conv2, outs)
        ctxt, un = self.forward_convbn_par_fhe(bb.conv2,
                                        bb.bn2, ctxt, ins)
        # Shortcut
        if len(bb.shortcut) > 0:
            convl, bnl = bb.shortcut
            _, ins_, _ = get_conv_params(convl, outs_in)
            shortcut, _ = self.forward_convbn_par_fhe(convl, bnl, ctxt_in, ins_, 
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

    ##############################

    def forward_convbn_par_fhe(self, cnn_layer, bn_layer, ctx, ins, kernels=[3,3]):
        U, ins, outs = get_conv_params(cnn_layer, ins)
        out = self.MultParConvBN_fhe(ctx, U, bn_layer, ins, outs, kernels)
        return out


    def SumSlots(self, ct_a,m,p):
        """Addition only"""
        ev = self.ev
        nrots = 0
        n = int(np.floor(np.log2(m)))
        ctx_b = [] ####
        ctx_b.append(ct_a) ####
        for j in range(1,n+1):
            lrots = int(p*2**(j-1))
            ctx_b.append(ev.add(ctx_b[j-1], 
                                ev.lrot(ctx_b[j-1], lrots, inplace=False),
                            inplace=False)) ####
            if lrots!=0:
                nrots=nrots+1  #______________________________ROTATION
        ctx_c = ctx_b[n] ####
        for j in range(0,n):
            n1 = np.floor((m/(2**j))%2)
            if n1==1:
                n2 =int(np.floor((m/(2**(j+1)))%2))
                lrots = int(p*2**(j+1))*n2
                ev.add(ctx_c, 
                    ev.lrot(ctx_b[j],lrots, inplace=False),
                    inplace=True) ####
                if lrots!=0:
                    nrots=nrots+1#____________________________ROTATION
        return ctx_c,nrots


    def MultParConvBN_fhe(self, ct_a, U, bn_layer, ins:Dict, outs:Dict,
                        kernels=[3,3],
                        nslots=2**15, 
                        scale_factor=1):
        ev = self.ev
        encoder = self.encoder
        hi,wi,ci,ki,ti,pi = [ins[k] for k in ins.keys()]
        ho,wo,co,ko,to,po = [outs[k] for k in outs.keys()]
        q = get_q(co,pi)
        fh,fw= kernels[0],kernels[1]
        print(f"[MultParConv] (hi,wi,ci,ki,ti,pi) =({hi:2},{wi:2},{ci:2},{ki:2},{ti:2}, {pi:2})")
        print(f"[MultParConv] (ho,wo,co,ko,to,po) =({ho:2},{wo:2},{co:2},{ko:2},{to:2}, {po:2})")
        print(f"[MultParConv] q = {q}")

        MuxBN_C, MuxBN_M, MuxBN_I = parMuxBN(bn_layer, outs, nslots)

        ct_d = self.gen_new_ctxt() ####
        ev.mod_down_by(ct_d, 2*ct_d.logp)
        ct = []
        nrots=0
        for i1 in range(fh):
            temp = []
            for i2 in range(fw):
                lrots = int((-(ki**2)*wi*(i1-(fh-1)/2) - ki*(i2-(fw-1)/2))) #both neg in the paper, git -,+
                temp.append(ev.lrot(ct_a, -lrots, inplace=False))
                if lrots!=0:
                    nrots = nrots+ 1#____________________________________ROTATION
            ct.append(temp)

        for i3 in range(q):
            ct_b = self.gen_new_ctxt() ####
            ev.mod_down_by(ct_b, ct_b.logp)
            for i1 in range(fh):
                for i2 in range(fw):
                    w = ParMultWgt(U,i1,i2,i3,ins,co,kernels,nslots)
                    w_enc = encoder.encode(w, ct[i1][i2].logp) ####
                    tmp = ev.mult_by_plain(ct[i1][i2], w_enc, inplace=False)
                    ev.rescale_next(tmp)
                    ev.add(ct_b, tmp, inplace=True) ####
            
            ct_c,nrots0 = self.SumSlots(ct_b, ki,              1)
            ct_c,nrots1 = self.SumSlots(ct_c, ki,          ki*wi)
            ct_c,nrots2 = self.SumSlots(ct_c, ti,  (ki**2)*hi*wi)
            nrots += nrots0 + nrots1 + nrots2#____________________________________ROTATION

            
            for i4 in range(0,min(pi,co-pi*i3)):
                i = pi*i3 +i4
                r0 = int(np.floor(nslots/pi))*(i%pi)
                r1 = int(np.floor(i/(ko**2)))*ko**2*ho*wo
                r2 = int(np.floor((i%(ko**2))/ko))*ko*wo
                r3 = i%ko
                rrots = (-r1-r2-r3)+r0
                rolled = ev.lrot(ct_c, rrots, inplace=False)
                S_mp = tensor_multiplexed_selecting(ho,wo,co,ko,to,i)
                vec_S = Vec(S_mp,nslots)
                tmp = ev.mult_by_plain(rolled, 
                                    encoder.encode(vec_S * MuxBN_C, 
                                                    rolled.logp), 
                                    inplace=False)
                ev.rescale_next(tmp)
                ev.add(ct_d, tmp, inplace=True)
                if rrots!=0:
                    nrots=nrots+1 #_________________________________________ROTATION

        for j in range(int(np.round(np.log2(po)))):
            r = -int(np.round(2**j*(nslots/po)))
            ev.add(ct_d, ev.lrot(ct_d, r, inplace=False), inplace=True)
            if r !=0:
                nrots+=1
        
        plain_vec = encoder.encode(-1/scale_factor*(MuxBN_C*MuxBN_M-MuxBN_I), 
                                ct_d.logp)
        ev.add_plain(ct_d, plain_vec, inplace=True)

        return ct_d
    
    def gen_new_ctxt(self):
        ev = self.ev
        this_contxt = ev.context.params
        newctxt = CiphertextStat(logp = this_contxt.logp,
                            logq = this_contxt.logq,
                            logn = this_contxt.logn)
        newctxt._enckey_hash = key_hash(ev.context.enc_key)
        return newctxt