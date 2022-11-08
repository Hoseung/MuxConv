from hecnn import *
from utils import *
from typing import Dict

##################################
#          MultParPack()         #
##################################

def MultParPack(A,dims=[],nslots=2**15):#image original
    ha,wa,ca,ka,ta,pa = [dims[k] for k in dims.keys()]
    A_mp = MultPack(A,dims,nslots)
    
    out = np.zeros(nslots)
    if pa !=int(pa):
        print(f"p is not an integer!!!!! p={pa}")
        return
    else:
        for i in range(0,pa):
            rot =-int(np.round(i*(nslots/pa)))
            out = out+ np.roll(A_mp,rot)
        return out

        
##################################
#          ParMultWgt()          #
##################################

def tensor_multiplexed_shifted_weight_par(U,i1,i2,i3,ins:Dict,outs:Dict):
    fh,fw= 3,3
    hi,wi,ci,ki,ti,pi =  [ins[k] for k in ins.keys()]
    ho,wo,co,ko,to,po = [outs[k] for k in outs.keys()]
    
    out = np.zeros([hi*ki, wi*ki,ti*pi])
    for i5 in range(hi*ki):
        for i6 in range(wi*ki):
            for i7 in range(ti*pi):     
                cond1 = ki**2*(i7%ti) + ki*(i5%ki) + (i6%ki)
                cond2 = np.floor(i5/ki)-(fh-1)/2+i1
                cond3 = np.floor(i6/ki)-(fw-1)/2+i2
                cond0 = np.floor(i7/ti)+pi*i3
                
                if ( cond0 >= co or 
                     cond1 >= ci or
                     cond2 not in range(hi) or 
                     cond3 not in range(wi)    ) :
                    out[i5][i6][i7] = 0
                else:
                    idx_3rd = ki**2*(i7%ti)+ki*(i5%ki)+i6%ki
                    idx_4th = int(np.floor(i7/ti)+pi*i3)
                    out[i5][i6][i7] = U[i1][i2][idx_3rd][idx_4th]
    return out

def ParMultWgt(U,i1,i2,i3,ins:Dict,outs:Dict,nslots=2**15):
    u = tensor_multiplexed_shifted_weight_par(U,i1,i2,i3,ins,outs)
    out = Vec(u,nslots)
    temp_size = int(nslots/ins['p'])    
    result = []
    for out_channel in range(ins['p']):
        temp = np.zeros(temp_size)
        for i in range (temp_size):
            temp[i]=out[out_channel*ins['h']*ins['w']*ins['k']**2*ins['c']+i]
        result+=list(temp)
    result = np.array(result)
    #print(result.shape)
    return result


##################################
#          MulParConv()          #
##################################

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
            #temp.append(np.roll(ct_a,-lrots))#1101
            if lrots!=0:
                nrots = nrots+ 1#____________________________________ROTATION
        ct.append(temp)
    for i3 in range(q):
        ct_b = np.zeros(nslots)
        for i1 in range(fh):
            for i2 in range(fw):
                w = ParMultWgt(U,i1,i2,i3,ins,outs,nslots)
                ct_b = ct_b + ct[i1][i2]*w                             
        ct_c,nrots0 = SumSlots(ct_b, ki,              1 )
        ct_c,nrots1 = SumSlots(ct_c, ki,          ki*wi )
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
    print(f"nrots_par={nrots}")
    return ct_d

