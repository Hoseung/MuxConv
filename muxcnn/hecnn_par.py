from muxcnn.hecnn import *
from muxcnn.utils import *
from typing import Dict
from math import floor

def MultParPack(A,dims=[],nslots=2**15):
    ha,wa,ca,ka,ta,pa = [dims[k] for k in dims.keys()]
    A_mp = MultPack(A,dims,nslots)
    
    out = np.zeros(nslots)
    if pa !=int(pa):
        print(f"p is not an integer!!!!! p={pa}")
        return
    else:
        for i in range(0,pa):
            rot = int(np.round(i*(nslots/pa)))
            out = out+ np.roll(A_mp,-rot)
        return out

def tensor_multiplexed_shifted_weight_par(U,i1,i2,i3,ins:Dict,co, kernels):
    fh,fw= kernels
    hi,wi,ci,ki,ti,pi =  [ins[k] for k in ins.keys()]
    
    fh_1_2_i1 = -(fh-1)/2+i1
    fh_1_2_i2 = -(fw-1)/2+i2
    pii3 = pi*i3
    out = np.zeros([hi*ki, wi*ki,ti*pi])
    range_hi = range(hi) # this is faster than np.arange(hi)
    range_wi = range(wi)
    for i5 in range(hi*ki):
        for i6 in range(wi*ki):
            for i7 in range(ti*pi):     
                cond1 = ki**2*(i7%ti) + ki*(i5%ki) + (i6%ki)
                cond2 = floor(i5/ki)+fh_1_2_i1
                cond3 = floor(i6/ki)+fh_1_2_i2
                cond0 = floor(i7/ti)+pii3
                
                if ( cond0 >= co or 
                     cond1 >= ci or
                     cond2 not in range_hi or 
                     cond3 not in range_wi    ) :
                    out[i5][i6][i7] = 0
                else:
                    #idx_3rd = ki**2*(i7%ti)+ki*(i5%ki)+i6%ki
                    #idx_4th = int(np.floor(i7/ti)+pi*i3)
                    out[i5][i6][i7] = U[i1][i2][cond1][cond0]
    return out

def ParMultWgt(U,i1,i2,i3,ins:Dict,co,kernels,nslots=2**15):
    u = tensor_multiplexed_shifted_weight_par(U,i1,i2,i3,ins,co,kernels)
    out = Vec(u,nslots)
    temp_size = int(nslots/ins['p'])
    result = np.zeros(nslots)
    h_w_c = ins['h']*ins['w']*ins['c']
    for out_channel in range(ins['p']):
        result[out_channel*temp_size:(out_channel+1)*temp_size] = out[out_channel*h_w_c:out_channel*h_w_c+temp_size]
    return result

def forward_conv_par(layer, ctx, ins):
    U, ins, outs = get_conv_params(layer, ins)    
    out = MultParConv(ctx, U, ins, outs)
    un = unpack(out,outs)
    return out, un


def MultParConv(ct_a,U,ins:Dict,outs:Dict,kernels=[3,3],nslots=2**15):
    hi,wi,ci,ki,ti,pi = [ins[k] for k in ins.keys()]
    ho,wo,co,ko,to,po = [outs[k] for k in outs.keys()]
    q = get_q(co,pi)
    fh,fw= kernels[0],kernels[1]
    print(f"[MultParConv] (hi,wi,ci,ki,ti,pi) =({hi:2},{wi:2},{ci:2},{ki:2},{ti:2}, {pi:2})")
    print(f"[MultParConv] (ho,wo,co,ko,to,po) =({ho:2},{wo:2},{co:2},{ko:2},{to:2}, {po:2})")
    print(f"[MultParConv] q = {q}")

    ct_d = np.zeros(nslots)
    ct = []
    nrots=0
    for i1 in range(fh):
        temp = []
        for i2 in range(fw):
            lrots = int((-(ki**2)*wi*(i1-(fh-1)/2) - ki*(i2-(fw-1)/2))) #both neg in the paper, git -,+
            temp.append(np.roll(ct_a,lrots))
            
            if lrots!=0:
                nrots = nrots+ 1#____________________________________ROTATION
        ct.append(temp)
    for i3 in range(q):
        ct_b = np.zeros(nslots)
        for i1 in range(fh):
            for i2 in range(fw):
                w = ParMultWgt(U,i1,i2,i3,ins,outs,kernels,nslots)
                ct_b = ct_b + ct[i1][i2]*w
        
        ct_c,nrots0 = SumSlots(ct_b, ki,              1)
        ct_c,nrots1 = SumSlots(ct_c, ki,          ki*wi)
        ct_c,nrots2 = SumSlots(ct_c, ti,  (ki**2)*hi*wi)
        nrots += nrots0 + nrots1 + nrots2#____________________________________ROTATION

        for i4 in range(0,min(pi,co-pi*i3)):
            i = pi*i3 +i4
            r0 = int(np.floor(nslots/pi))*(i%pi)
            r1 = int(np.floor(i/(ko**2)))*ko**2*ho*wo
            r2 = int(np.floor((i%(ko**2))/ko))*ko*wo
            r3 = i%ko
            rrots = (-r1-r2-r3)+r0
            rolled =  np.roll(ct_c, -rrots)
            S_mp = tensor_multiplexed_selecting(ho,wo,co,ko,to,i)
            vec_S = Vec(S_mp,nslots)
            ct_d = ct_d +rolled*vec_S
            if rrots!=0:
                nrots=nrots+1 #_________________________________________ROTATION
    for j in range(int(np.round(np.log2(po)))):
        r = int(np.round(2**j*(nslots/po)))
        ct_d = ct_d + np.roll(ct_d,r)
        if r !=0:
            nrots+=1
            #ic(po,j)

    return ct_d

###################################################################

def get_bn_params(bn_layer):
    bn_T = bn_layer.weight # Gamma
    bn_V = bn_layer.running_var
    bn_M = bn_layer.running_mean 
    bn_I = bn_layer.bias  # Beta
    bn_eps = bn_layer.eps
    return bn_T, bn_V, bn_M, bn_I, bn_eps
# As a function of i1, i2, i3 and constants nt, ki, hi, wi, ti, ci, and pi 

def mux_BN_const(H, ins, nslots):
    hi,wi,ci,ki,ti,pi = [ins[k] for k in ins.keys()]
    nt_over_pi = int(nslots/pi)
    ki2hiwi = ki**2*hi*wi
    npix_per_mult_ch_img = ki2hiwi*ti
    
    new_H = np.zeros(nslots)
    for j in range(nslots):
        j_mod_nt_over_pi = j%nt_over_pi
        i1 = floor((j_mod_nt_over_pi % ki2hiwi)/ (ki*wi))
        i2 = j_mod_nt_over_pi % (ki*wi)
        i3 = floor(j_mod_nt_over_pi / ki2hiwi)
        cond2 = ki**2*i3 + ki*(i1%ki) + i2%ki

        if (j_mod_nt_over_pi >= npix_per_mult_ch_img) or \
            (cond2 >= ci):
            new_H[j] = 0
        else:
            new_H[j] = H[cond2]
    
    return new_H

def parMuxBN(bn_layer, outs, nslots):
    bn_T, bn_V, bn_M, bn_I, bn_eps = get_bn_params(bn_layer)
    if bn_T is None:
        bn_T = np.ones_like(bn_V)
    if bn_I is None:
        bn_I = np.zeros_like(bn_V)
    bn_C = bn_T/np.sqrt(bn_V+bn_eps)
    expanded_C = mux_BN_const(bn_C, outs, nslots) # 
    expanded_M = mux_BN_const(bn_M, outs, nslots) # MEAN
    expanded_I = mux_BN_const(bn_I, outs, nslots) # BIAS

    return expanded_C, expanded_M, expanded_I

def MultParConvBN(ct_a, U, bn_layer, ins:Dict, outs:Dict,
                    kernels=[3,3],
                    nslots=2**15, 
                    scale_factor=1):
    hi,wi,ci,ki,ti,pi = [ins[k] for k in ins.keys()]
    ho,wo,co,ko,to,po = [outs[k] for k in outs.keys()]
    q = get_q(co,pi)
    fh,fw= kernels[0],kernels[1]
    print(f"[MultParConv] (hi,wi,ci,ki,ti,pi) =({hi:2},{wi:2},{ci:2},{ki:2},{ti:2}, {pi:2})")
    print(f"[MultParConv] (ho,wo,co,ko,to,po) =({ho:2},{wo:2},{co:2},{ko:2},{to:2}, {po:2})")
    print(f"[MultParConv] q = {q}")

    MuxBN_C, MuxBN_M, MuxBN_I = parMuxBN(bn_layer, outs, nslots)

    ct_d = np.zeros(nslots)
    ct = []
    nrots=0
    for i1 in range(fh):
        temp = []
        for i2 in range(fw):
            lrots = int((-(ki**2)*wi*(i1-(fh-1)/2) - ki*(i2-(fw-1)/2))) #both neg in the paper, git -,+
            temp.append(np.roll(ct_a,lrots))
            if lrots!=0:
                nrots = nrots+ 1#____________________________________ROTATION
        ct.append(temp)

    for i3 in range(q):
        ct_b = np.zeros(nslots)
        for i1 in range(fh):
            for i2 in range(fw):
                w = ParMultWgt(U,i1,i2,i3,ins,co,kernels,nslots)
                #print("w = w2?", np.array_equal(w,w2))
                ct_b = ct_b + ct[i1][i2]*w     
        #print(ct_b)#[2000:2010])                        
        ct_c,nrots0 = SumSlots(ct_b, ki,              1)
        ct_c,nrots1 = SumSlots(ct_c, ki,          ki*wi)
        ct_c,nrots2 = SumSlots(ct_c, ti,  (ki**2)*hi*wi)
        #print(ct_c)#[2000:2010])
        nrots += nrots0 + nrots1 + nrots2#____________________________________ROTATION
        
        for i4 in range(0,min(pi,co-pi*i3)):
            i = pi*i3 +i4
            r0 = int(np.floor(nslots/pi))*(i%pi)
            r1 = int(np.floor(i/(ko**2)))*ko**2*ho*wo
            r2 = int(np.floor((i%(ko**2))/ko))*ko*wo
            r3 = i%ko
            rrots = (-r1-r2-r3)+r0
            rolled = np.roll(ct_c, -rrots)
            S_mp = tensor_multiplexed_selecting(ho,wo,co,ko,to,i)
            vec_S = Vec(S_mp,nslots)
            #print("plain", (vec_S * MuxBN_C)[2040:2050])
            tmp = rolled*vec_S * MuxBN_C
            #print("tmp", tmp[2040:2050])
            ct_d += tmp
            
            if rrots!=0:
                nrots=nrots+1 #_________________________________________ROTATION

    #print(ct_d[2040:2050])#[2000:2010])
    for j in range(int(np.round(np.log2(po)))):
        r = int(np.round(2**j*(nslots/po)))
        ct_d += np.roll(ct_d,r)
        if r !=0:
            nrots+=1
            #ic(po,j)
    #print(ct_d[2040:2050])#[2000:2010])
    ct_d -= 1/scale_factor*(MuxBN_C*MuxBN_M-MuxBN_I)

    return ct_d

from time import time
def forward_convbn_par(cnn_layer, bn_layer, ctx, ins, kernels=[3,3]):
    t0 = time()
    U, ins, outs = get_conv_params(cnn_layer, ins)
    out = MultParConvBN(ctx, U, bn_layer, ins, outs, kernels)
    un = unpack(out,outs)
    print(f"{time() - t0:.4f} s")
    return out, un


def select_AVG(nslots, i3, ki):
    kii3 = ki*i3
    S_avg = np.zeros(nslots)
    S_avg[kii3:kii3+ki] = 1
        
    return S_avg 

def AVGPool(ct_in, ins, nslots):
    ct_a = ct_in.copy()
    ct_b = np.zeros(nslots)

    hi,wi,ci,ki,ti,pi = [ins[k] for k in ins.keys()]

    # 한 페이지에 분포하는 64개 숫자를 더함
    for j in range(int(np.log2(wi))):
        ct_a += np.roll(ct_a, -2**j*ki) # 4, 8, 16

    for j in range(int(np.log2(hi))):
        ct_a += np.roll(ct_a, -2**j*ki*ki*wi) # 128, 256, 512

    ### + 64채널에서 하나씩을 뽑아옴.
    for i1 in range(ki): # 4
        for i2 in range(ti): # 
            S_vec = select_AVG(nslots, ki*i2+i1, ki) / (hi*wi) # 4개 숫자만 추출
            ct_b += np.roll(ct_a, -(ki**2*hi*wi*i2 + ki*wi*i1 - ki*(ki*i2+i1)))* S_vec
            
    return ct_b

def _ParMultWgt_deprecated(U,i1,i2,i3,ins:Dict,outs:Dict,kernels,nslots=2**15):
    u = tensor_multiplexed_shifted_weight_par(U,i1,i2,i3,ins,outs,kernels)
    out = Vec(u,nslots)
    temp_size = int(nslots/ins['p'])    
    result = []
    
    for out_channel in range(ins['p']):
        temp = np.zeros(temp_size)
        for i in range (temp_size):
            temp[i]=out[out_channel*ins['h']*ins['w']*ins['c']+i] 
        result+=list(temp)
    result = np.array(result)
    return result
