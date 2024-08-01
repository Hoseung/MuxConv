import numpy as np
from math import ceil
import torch.nn as nn
from typing import Dict
from muxcnn.models import ResNet20
from .comparator_heaan import ApprRelu_HEAAN
# from hemul import loader 
# he = loader.load()
from fase.heaan_loader import load
he = load()
from muxcnn.utils import get_q, get_conv_params, get_channel_last
from muxcnn.hecnn_par import (MultParPack, 
                                parMuxBN, 
                                tensor_multiplexed_selecting, 
                                Vec, 
                                ParMultWgt)
from muxcnn.hecnn_par import select_AVG
from time import time

def relu(x):
    x[x<0] = 0
    return x
class ResNetHEAAN():
    def __init__(self, model, hec, 
                    alpha = 14, 
                    xmin=-40, 
                    xmax=40, 
                    min_depth=True, 
                    debug=False
                    ):
        self.torch_model = model
        self.torch_model.eval()
        self.hec = hec
        self.nslots = 2**hec.parms.logn
        self.alpha=alpha
        self.debug=debug
        self._set_activation(alpha=self.alpha, xmin=xmin, xmax=xmax, min_depth=min_depth)
        
    def _set_activation(self, eps=0.01, margin=0.0005, *args, **kwargs):        
        if self.debug:
            self._activation_org = relu
        else:
            # [0801] ApprRelu_HEAAN 대신 2차 Hermite Polynomial 사용. (ReLU 대신 2차함수)
            self.activation = ApprRelu_HEAAN(self.hec, eps=eps, margin=margin, *args, **kwargs)

    def forward(self, ctxt, ki=1, hi=32, wi=32, debug=None, verbose=True):
        # [0801] FHE 연산을 직접 사용하지 않는 부분이므로 수정 필요 없음 (Her 버전에서도 동일)
        if debug is None: debug = self.debug
        t0 = time()
        if verbose: print("[FHE_CNN] Inference started...")
        model = self.torch_model
        ctxt, outs0 = self.forward_early(ctxt, ki, hi, wi, debug=debug)
        #self.hec.rescale(ctxt)
        # Basic blocks
        t1 = time()
        ctxt, outs1 = self.forward_bb(model.layer1[0], ctxt, outs0, debug=debug)
        if verbose: print("[FHE_CNN] First Basic Block finished in {:.2f} sec\n".format(time()-t1))
        t1 = time()
        ctxt, outs2 = self.forward_bb(model.layer2[0], ctxt, outs1, debug=debug)
        if verbose: print("[FHE_CNN] Second Basic Block finished in {:.2f} sec\n".format(time()-t1))
        t1 = time()
        ctxt, outs3 = self.forward_bb(model.layer3[0], ctxt, outs2, debug=debug)
        if verbose: print("[FHE_CNN] Third Basic Block finished in {:.2f} sec\n".format(time()-t1))
        t1 = time()
        ctxt = self.AVGPool(ctxt, outs3, self.nslots) # Gloval pooling
        if debug: 
            print("[After AVGPool", self.hec.decrypt(ctxt))
        if verbose: print("[FHE_CNN] Global AVGPool finished in {:.2f} sec\n".format(time()-t1))
        t1 = time()
        result = self.forward_linear(ctxt, model.linear)
        if debug: 
            print("[After Linear", self.hec.decrypt(result))
        if verbose: print("[FHE_CNN] FullyConnected finished in {:.2f} sec\n".format(time()-t1))
        if verbose: print("[FHE_CNN] Inference finished in {:.2f} sec\n".format(time()-t0))
        return result

    def pack_img_ctxt(self, img_tensor):
        # [0801] encrypt하기 전까진 CTXT가 아니므로 FHE backend와 무관
        """ For convenience
        """
        model = self.torch_model
        imgl = get_channel_last(img_tensor[0].detach().numpy())
        ki = 1 # initial ki
        hi, wi, ch = imgl.shape

        # early conv and bn
        _, ins0, outs0 = get_conv_params(model.conv1, {'k':ki, 'h':hi, 'w':wi})
        ct_a = MultParPack(imgl, ins0)
        # [0801] encrypt를 OFHE로 변경
        return self.hec.encrypt(ct_a)

    def forward_early(self, ct_a, ki, hi, wi, debug=False, verbose=True):
        # [0801] 여기도 debug부분 말고는 모두 high-level이어서 FHE 백엔드와 무관
        model = self.torch_model
        _, ins0, outs0 = get_conv_params(model.conv1, {'k':ki, 'h':hi, 'w':wi})

        t0 = time()
        if verbose: print("[FHE_CNN_EARLY] ConvBN started...")
        ctxt = self.forward_convbn_par_fhe(model.conv1, 
                                        model.bn1, ct_a, ins0)
        if verbose: print("[FHE_CNN_EARLY] ConvBN finished in {:.2f} sec".format(time()-t0))

        t0 = time()
        if verbose: print("[FHE_CNN_EARLY] ReLU started...")
        if debug: 
            tmp = self.hec.decrypt(ctxt)
            activated = self._activation_org(tmp)
            ctxt = self.hec.encrypt(activated)
        else:
            ctxt = self.activation(ctxt)
        if verbose: print("[FHE_CNN_EARLY] ReLU hinished in {:.2f} sec".format(time()-t0))
        return ctxt, outs0 

    def forward_bb(self, bb:ResNet20.BasicBlock, ctxt_in, outs_in, debug=False, verbose=True):
        # Bootstrap before shortcut
        if ctxt_in.logq <= 80:
            ctxt_in = self.hec.bootstrap2(ctxt_in) # [0801] Ofhe에서는 bootstrapping 자동 모드가 있으므로 HEAAN의 bootstrapping은 모두 무시
            if debug: print("MuxBN bootstrap", ctxt_in.logp, ctxt_in.logq)

        shortcut = he.Ciphertext(ctxt_in) # [0801] 원래는 self.hec.copy(ctxt_in)이어야 올바름.

        _, ins, outs = get_conv_params(bb.conv1, outs_in) # [0801] FHE backend와 무관 -- 파라미터 모양이 복잡 -> Lee.2022의 알고리즘
        t0 = time() 
        if verbose: print("[FHE_CNN BasicBlock] ConvBN1 started...")
        ctxt = self.forward_convbn_par_fhe(bb.conv1,
                                        bb.bn1, ctxt_in, ins) # [0801] 들어가면 FHE계산 있음. function signature는 그대로 유지해도 됨
        if verbose: print("[FHE_CNN BasicBlock] ConvBN1 finished in {:.2f} sec".format(time()-t0))
        t0 = time()
        if verbose: print("[FHE_CNN BasicBlock] ReLU1 started...")
        if debug: 
            tmp = self.hec.decrypt(ctxt)
            activated = self._activation_org(tmp)
            ctxt = self.hec.encrypt(activated)
        else:
            ctxt = self.activation(ctxt) # [0801] 통째로 Her 버전으로 대체
        #ctxt = self.activation(ctxt)
        if verbose: print("[FHE_CNN BasicBlock] ReLU1 finished in {:.2f} sec".format(time()-t0))
        if debug: 
            print("After activation", ctxt)
            print("FHE result:", self.hec.decrypt(ctxt)[:100])
        _, ins, outs = get_conv_params(bb.conv2, outs)
        t0 = time()
        if verbose: print("[FHE_CNN BasicBlock] ConvBN2 started...")
        ctxt = self.forward_convbn_par_fhe(bb.conv2,
                                        bb.bn2, ctxt, ins)
        if verbose: print("[FHE_CNN BasicBlock] ConvBN2 finished in {:.2f} sec".format(time()-t0))    
        #pickle.dump(self.hec.decrypt(ctxt), open("ctxt3.pkl", "wb"))
        if debug:
            print("\n\n ddddddd")
            print(self.hec.decrypt(ctxt))
        # Shortcut
        if len(bb.shortcut) > 0:
            convl, bnl = bb.shortcut
            _, ins_, _ = get_conv_params(convl, outs_in)
            t0 = time()
            if verbose: print("[FHE_CNN BasicBlock] ConvBN_shortcut started...")
            shortcut = self.forward_convbn_par_fhe(convl, bnl, shortcut, ins_, 
                                                convl.kernel_size)
            if verbose: print("[FHE_CNN BasicBlock] ConvBN_shortcut finished in {:.2f} sec".format(time()-t0))
        #pickle.dump(self.hec.decrypt(shortcut), open("shortcut.pkl", "wb"))
        if debug:
            print("[forward_bb] Shortcut", shortcut)
            print("ctxt", ctxt)

        # [0801] HEAAN에서는 CTXT의 스케일을 자동으로 관리해주지 않아서 다른 루트로 온 CTXT를 더하기 전에
        # 스케일 맞춰주는 과정이 필요.  (rescale, mat_mod_
        # OFHE에서는 신경 안 써도 됨. 
        # Add shortcut
        if ctxt.logp > shortcut.logp:
            if debug: print("ctxt > shortcut")
            self.hec.rescale(ctxt, shortcut.logp)
        elif ctxt.logp < shortcut.logp:
            if debug: print("shortcut > ctxt")
            self.hec.rescale(shortcut, ctxt.logp)
        
        if ctxt.logq > shortcut.logq:
            self.hec.match_mod(ctxt, shortcut)
        elif ctxt.logq < shortcut.logq:
            self.hec.match_mod(shortcut, ctxt)
        if debug: 
            print(ctxt, shortcut)
            print(self.hec.decrypt(ctxt))
            print(self.hec.decrypt(shortcut))
        
        # [0801] OFHE버전의 'wrapper'에 add를 구현 하면 됨.
        # [0801] Add나 Mult의 inplace 옵션을 유지하는게 유리함. 
        self.hec.add(ctxt, shortcut, inplace=True) 
        #pickle.dump(self.hec.decrypt(ctxt), open("ctxt4.pkl", "wb"))
        #ctxt += shortcut
        t0 = time()
        if verbose: print("[FHE_CNN BasicBlock] ReLU2 started...")
        # Activation
        if debug: 
            tmp = self.hec.decrypt(ctxt) 
            activated = self._activation_org(tmp)
            ctxt = self.hec.encrypt(activated)
        else:
            ctxt = self.activation(ctxt)
        if verbose: print("[FHE_CNN BasicBlock] ReLU2 finished in {:.2f} sec".format(time()-t0))

        #pickle.dump(self.hec.decrypt(ctxt), open("ctxt5.pkl", "wb"))

        return ctxt, outs

    def forward_linear(self, ctxt, linearl:nn.modules.Linear, verbose=True):
        hec = self.hec
        no, ni = linearl.weight.shape

        weight_vec = np.zeros(self.nslots)
        weight_vec[:no*ni] = np.ravel(linearl.weight.detach().numpy())

        # make 10 copies of 64 flattened values
        for i in range(ceil(np.log2(no))):
            #ctxt += np.roll(ctxt, 2**i*ni)
            hec.add(ctxt, 
                hec.lrot(ctxt, -2**i*ni, inplace=False),
                        inplace=True) # [0801] Rotation도 OFHE버전의 wrapper에 구현
            

        # multiply 64 * 10 at once
        # AVGPool에서 S_vec를 조절하면 이 단계의 일부를 미리 수행할 수 있음 !!
        #ctxt = weight_vec * ctxt
        # [0801] ctxt * np.array 연산
        hec.multByVec(ctxt, weight_vec, inplace=True) 
        # [0801] OFHE버전의 wrapper에 구현
        # [0801] 어쩌면 OFHE에서 Polymorphism을 지원해서, 그냥 ndarray로 넣어도 알아서 처리할 수 있을지도.
        hec.rescale(ctxt) # [0801] Rescale은 무시

        # Sum 64 numbers each 
        for j in range(int(np.log2(ni))):
            #ctxt += np.roll(ctxt, -2**j)
            hec.add(ctxt, 
                hec.lrot(ctxt, 2**j, inplace=False),
                        inplace=True)

        return ctxt

    def __call__(self, img_tensor):
        return self.forward(img_tensor)

    ##############################
    def forward_convbn_par_fhe(self, cnn_layer, bn_layer, ctx, ins, kernels=[3,3]):
        U, ins, outs = get_conv_params(cnn_layer, ins)
        return self.MultParConvBN_fhe(ctx, U, bn_layer, ins, outs, kernels)


    def SumSlots(self, ct_a,m,p):
        # FHE_Simulator (hemul)에서 Numpy버전 참고하여 OFHE로 작성
        """Addition only"""
        ev = self.hec
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
                        scale_factor=1, debug=False):
        """Consumes two mults"""
        if ct_a.logq <= 80:
            ct_a = self.hec.bootstrap2(ct_a) #[0801] 무시
        ev = self.hec # [0801] 이 부분 기억!

        #encoder = self.hec
        hi,wi,ci,ki,ti,pi = [ins[k] for k in ins.keys()]
        ho,wo,co,ko,to,po = [outs[k] for k in outs.keys()]
        q = get_q(co,pi)
        fh,fw= kernels[0],kernels[1]
        print(f"[MultParConvBN] Layer structure: (hi,wi,ci,ki,ti,pi) =({hi:2},{wi:2},{ci:2},{ki:2},{ti:2}, {pi:2})")

        MuxBN_C, MuxBN_M, MuxBN_I = parMuxBN(bn_layer, outs, nslots)

        ct_d = self.gen_new_ctxt() ####
        
        ev.modDownTo(ct_d, ct_a.logq - 2*ct_d.logp) # [0801] OFHE에서는 무시
        if debug: print("ct_d", ct_d.logp, ct_d.logq)
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
            ev.modDownTo(ct_b, ct[0][0].logq - ct_b.logp)

            for i1 in range(fh):
                for i2 in range(fw):
                    w = ParMultWgt(U,i1,i2,i3,ins,co,kernels,nslots)
                    tmp = ev.multByVec(ct[i1][i2], w, inplace=False)
                    ev.rescale(tmp)
                    ev.add(ct_b, tmp, inplace=True) # [0801] ev = HEAAN.evaluator == self.hec랑 같음.
            
            ct_c,nrots0 = self.SumSlots(ct_b, ki,              1) #[0801] numpy버전 참고해서 OFHE로 작성되어있을 것으로..
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
                tmp = ev.multByVec(rolled, vec_S * MuxBN_C, 
                                                    #rolled.logp), 
                                    inplace=False)
                ev.rescale(tmp)
                ev.add(ct_d, tmp, inplace=True)
                if rrots!=0:
                    nrots=nrots+1 #_________________________________________ROTATION

        for j in range(int(np.round(np.log2(po)))):
            r = -int(np.round(2**j*(nslots/po)))
            ev.add(ct_d, ev.lrot(ct_d, r, inplace=False), inplace=True)
            if r !=0:
                nrots+=1
        
        plain_vec = -1/scale_factor*(MuxBN_C*MuxBN_M-MuxBN_I)
                                        #,ct_d.logp)
        ev.addConst(ct_d, plain_vec, inplace=True) 
        # [0801] multByVen처럼 OFHE에서는 바로 지원할 수도 있음.
        # [0801] 말하자면 add Vec임. 


        return ct_d
    
    def gen_new_ctxt(self):
        parms = self.hec.parms
        return he.Ciphertext(parms.logp, parms.logq, parms.n)

    def AVGPool(self, ct_in, ins, nslots, verbose=True):
        hec = self.hec
        ct_a = he.Ciphertext(ct_in) # [0801] 수정 필요 -> hec.copy()
        #ct_b = np.zeros(nslots)
        ct_b = self.gen_new_ctxt() # [0801] hec.new_ctxt()? 
        hi,wi,ci,ki,ti,pi = [ins[k] for k in ins.keys()]

        # 한 페이지에 분포하는 64개 숫자를 더함
        for j in range(int(np.log2(wi))):
            # [0801] forward_linear에서 이미 해결했음
            hec.add(ct_a, 
                hec.lrot(ct_a, 2**j*ki, inplace=False),
                        inplace=True)
            #ct_a += np.roll(ct_a, -2**j*ki) # 4, 8, 16

        for j in range(int(np.log2(hi))):
            #ct_a += np.roll(ct_a, -2**j*ki*ki*wi) # 128, 256, 512
            hec.add(ct_a, 
                hec.lrot(ct_a, 2**j*ki*ki*wi, inplace=False),
                        inplace=True) 

        # [0801] OFHE에서는 무시
        hec.modDownTo(ct_b, ct_a.logq - ct_a.logp)
        ### + 64채널에서 하나씩을 뽑아옴.
        for i1 in range(ki): # 4
            for i2 in range(ti): # 
                S_vec = select_AVG(nslots, ki*i2+i1, ki) / (hi*wi) # 4개 숫자만 추출
                #ct_b += np.roll(ct_a, -(ki**2*hi*wi*i2 + ki*wi*i1 - ki*(ki*i2+i1)))* S_vec
                tmp = hec.lrot(ct_a, (ki**2*hi*wi*i2 + ki*wi*i1 - ki*(ki*i2+i1)), inplace=False)
                hec.multByVec(tmp, S_vec, inplace=True)
                hec.rescale(tmp) # [0801] 무시
                hec.add(ct_b, tmp, inplace=True) 
                
        return ct_b

