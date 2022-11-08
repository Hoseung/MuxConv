from typing import List, Dict
from muxcnn.utils import *

def Vec(mat,nslots):
    hi,wi,ci = np.shape(mat)
    out = np.zeros(nslots)
    for i in range(hi*wi*ci):
        idx_1st = int(np.floor((i%(hi*wi))/wi))
        idx_2nd = i%wi
        idx_3rd = int(np.floor(i/(hi*wi)))
        out[i]=mat[idx_1st][idx_2nd][idx_3rd]
    return out

def tensor_multiplexed_input(mat,dims=[]):
    hi,wi,ci, ki,ti,pi = [dims[k] for k in dims.keys()]
    out = np.zeros([ki*hi,ki*wi,ti])
    for i3 in range(ki*hi):
        for i4 in range(ki*wi):
            for i5 in range(ti):
                cond = (ki**2)*i5 + ki*(i3%ki) + (i4%ki)
                if cond<ci:
                    idx_1st = int(np.floor(i3/ki))
                    idx_2nd = int(np.floor(i4/ki))
                    idx_3rd = (ki**2)*i5 + ki*(i3%ki) + i4%ki
                    out[i3][i4][i5] = mat[idx_1st][idx_2nd][idx_3rd]
    return out

def MultPack(mat,dims=[],nslots=2**15):
    return Vec(tensor_multiplexed_input(mat,dims),nslots)

def unpack(ct,dims=[]):#제대로 작동(10.31) 
    ha,wa,ca,ka,ta,pa = [dims[k] for k in dims.keys()]
    tsize = ha*wa*ka**2
    ct = ct[:ta*tsize]
    ch  = []
    nMultChs = ta*ka**2 
    for channel in range(nMultChs):
        r = channel%(ka**2)
        idx_start_channel = int(np.floor(channel/ka**2))*tsize + int(np.floor(r/ka))*wa*ka + (r%ka)
        mat=[]
        for i in range(ha):
            row = []
            for j in range(wa):
                idx = idx_start_channel+i*wa*ka**2+j*ka
                row.append(ct[idx])
            mat.append(row)
        ch.append(mat)
    return np.array(ch)

def tensor_multiplexed_selecting(ho,wo,co,ko,to,i):
    S = np.zeros([ko*ho, ko*wo,to])
    for i3 in range (ko*ho):
        for i4 in range (ko*wo):
            for i5 in range (to):
                cond = i5*ko**2 + ko*(i3%ko) + (i4%ko)
                if cond==i:
                    S[i3,i4,i5] = 1
    return S

def tensor_multiplexed_shifted_weight(U,i1,i2,i,ins:Dict):
    hi,wi,ci,ki,ti,pi = [ins[k] for k in ins.keys()]
    fh,fw= 3,3
    out = np.zeros([hi*ki, wi*ki,ti])
    for i3 in range(hi*ki):
        for i4 in range(wi*ki):
            for i5 in range(ti):
                cond1 = ki**2*i5 + ki*(i3%ki) + i4%ki
                cond2 = np.floor(i3/ki)-(fh-1)/2+i1
                cond3 = np.floor(i4/ki)-(fw-1)/2+i2
                if (cond1 >= ci or cond2 not in range(hi) or cond3 not in range(wi)):
                    out[i3][i4][i5] = 0
                else:
                    out[i3][i4][i5] = U[i1][i2][ki**2*i5+ki*(i3%ki)+i4%ki][i]
    return out

def MultWgt(U,i1,i2,i,ins=[],nslots=2**15):
    out = np.zeros(nslots)
    temp = Vec(tensor_multiplexed_shifted_weight(U,i1,i2,i,ins),nslots)
    out[:temp.size]=temp
    return out

def MultConv(ct_a,U,ins:Dict,outs:Dict,kernels=[3,3],nslots=2**15):
    hi,wi,ci,ki,ti,pi = [ins[k] for k in ins.keys()]
    ho,wo,co,ko,to,po = [outs[k] for k in outs.keys()]
    fh,fw= kernels[0],kernels[1]
    print(f"[MultConv] (hi,wi,ci) =({hi},{wi},{ci}),(ho,wo,co)=({ho},{wo},{co}),(fh,fw)=({fh},{fw})")
    print(f"[MultConv] (ki,ti) =({ki},{ti}), (ko,to) =({ko},{to})")
    ct_d = np.zeros(nslots)
    ct = []
    nrots=0
    for i1 in range(fh):
        temp = []
        for i2 in range(fw):
            lrots = int((-(ki**2)*wi*(i1-(fh-1)/2) - ki*(i2-(fw-1)/2))) #both neg in the paper, git -,+
            if lrots!=0:
                temp.append(np.roll(ct_a,lrots))
                nrots = nrots+ 1#____________________________________ROTATION
            else:
                temp.append(ct_a)
        ct.append(temp)
    for i in range(co):
        ct_b = np.zeros(nslots)
        for i1 in range(fh):
            for i2 in range(fw):
                w = MultWgt(U,i1,i2,i,ins,nslots)
                ct_b = ct_b + ct[i1][i2]*w
        ct_c,nrots0 = SumSlots(ct_b, ki,              1 )
        ct_c,nrots1 = SumSlots(ct_c, ki,          ki*wi )
        ct_c,nrots2 = SumSlots(ct_c, ti,  (ki**2)*hi*wi )
        nrots += nrots0 + nrots1 + nrots2
        
        r1 =  int(np.floor(i/(ko**2)))*ko**2*(ho)*(wo)
        r2 =  int( np.floor((i%(ko**2))/ko))*ko*(wo)
        r3 =  (i%ko)
        rrots = -r1-r2-r3
        rolled =  np.roll(ct_c, -rrots)
        S_mp = tensor_multiplexed_selecting(ho,wo,co,ko,to,i)
        vec_S = Vec(S_mp,nslots)
               
        ct_d = ct_d +rolled*vec_S
        if rrots!=0:
            nrots=nrots+1 #_________________________________________ROTATION
    print(f"nrots_par={nrots}")
    return ct_d


def forward_conv(layer, ctx, ki, hi, wi):
    U, ins, outs = get_conv_params(layer, ki=ki, hi=hi, wi=wi)    
    out = MultConv(ctx, U, ins, outs)
    un = unpack(out,outs)
    return out, un
