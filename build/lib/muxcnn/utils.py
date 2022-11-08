import numpy as np
from matplotlib import pyplot as plt
from math import ceil, floor
from PIL import Image
import cv2
from icecream import ic
#############
#   Utils   #
#############

def imshow_with_value(data,pt):
    if (len(data.shape))==3:
        h,w,c = data.shape
        channel = {'R':0,'G':1,'B':2}
    else:
        h,w = data.shape
    fig, ax = plt.subplots()
    im = ax.imshow(data)
    # Loop over data dimensions and create text annotations.
    for i in range(h):
        for j in range(w):
            if (len(data.shape)==3):
                if pt==0:
                    text = ax.text(j, i, int(data[i, j, channel['R']]), ha="center", va="center", color="w")
                else:
                    text = ax.text(j, i, np.round(data[i, j, channel['R']],pt), ha="center", va="center", color="w")
            else:
                if pt==0:
                    text = ax.text(j, i, int(data[i, j]), ha="center", va="center", color="w")  
                else:
                    text = ax.text(j, i, np.round(data[i, j],pt), ha="center", va="center", color="w")  
                
def calculate_nrots(fh,fw,co,ki,ti):
    return fh*fw-1 + co*(2*np.floor(np.log2(ki)) +np.floor(np.log2(ti))+ 1) 

def count_ones(mat):
    count=0
    h,w = mat.shape
    for i in range(h):
        for j in range(w):
            if mat[i,j]==1:
                count=count+1
    return count

def get_sparse_matrix(h,w,m,p):
    out = np.zeros([h,w])
    for i in range(0,h,p):
        for j in range(0,w,p):
            for k in range(m):
                out[i,j+k] = 1 
    return out

def get_p(nslots,h,w,k,t):
    exp = np.floor(np.log2(nslots/(h*w*t*k**2)))
    return int(2**exp)

def get_q(co,pi):
    return ceil(co/pi)

def get_t(c,k):
    return ceil(c/k**2)

def get_channel_first(U):
    dim=U.shape
    if len(dim)==4:
        h,w,ci,co = U.shape
        out = np.zeros([co,ci,h,w])
        for d in range(co):
            for z in range(ci):
                for i in range(h):
                    for j in range(w):
                        out[d,z,i,j]=U[i,j,z,d]
    elif len(dim)==3:
        h,w,c = U.shape
        out = np.zeros([c,h,w])
        for z in range(c):
            for i in range(h):
                for j in range(w):
                    out[z,i,j] = U[i,j,z]
    return out

def get_channel_last(U):
    """
    엄청 느릴 것 같은데..."""
    dim=U.shape
    if len(dim)==4:
        co,ci,h,w = U.shape
        out = np.zeros([h,w,ci,co])
        for d in range(co):
            for z in range(ci):
                for i in range(h):
                    for j in range(w):
                        out[i,j,z,d]=U[d,z,i,j]
    elif len(dim)==3:
        c,h,w = U.shape
        out = np.zeros([h,w,c])
        for z in range(c):
            for i in range(h):
                for j in range(w):
                    out[i,j,z] = U[z,i,j]
    return out


#############################
#  kernels that do nothing  #
#############################
def create_U(fh,fw,ci,co):
    U = np.zeros([fh,fw,ci,co])
    for l in range(co):
        for k in range(ci):
            for i in range(fh):
                for j in range(fw):
                    if i==1 and j==1:
                        U[i,j,k,l]=1
    #ic(U.shape)
    return U
def create_U_select(fh,fw,ci,co,kname):
    U = np.zeros([fh,fw,ci,co])
    sharpen=np.array([[ 0,-1, 0],[-1, 5,-1],[ 0,-1, 0]])
    dummy=np.array([[ 0, 0, 0],[0, 1, 0],[ 0, 0, 0]])
    blur=np.dot(np.array([[ 1, 1, 1],[1, 1, 1],[ 1, 1, 1]]),1/9)
    select = {"sharpen":sharpen,"dummy":dummy,"blur":blur}
    kernel = select[kname]
    for l in range(co):
        for k in range(ci):
            for i in range(fh):
                for j in range(fw):
                    U[i,j,k,l]=kernel[i,j]
    #ic(U.shape)
    return U

#############################
#  image                    #
#############################
def create_img(ins=[],isFromFile=True):
    hi,wi,ci,ki,ti,pi = ins[0],ins[1],ins[2],ins[3],ins[4],ins[5]
    if isFromFile is True:
        img = cv2.resize(cv2.imread("cute.jpg"),(hi,wi))
        temp = list(get_channel_first(img))
        while(1):
            if len(temp)>=ci:
                break
            temp.append(np.ones([hi,wi]))
        #ic(np.array(temp).shape)
        img = get_channel_last(np.array(temp))
    else:
        img = np.zeros([hi,wi,ci])
        for i in range(hi):
            for j in range(wi):
                for k in range(ci):
                    img[i,j,k] = 100*(i+1)+10*(j+1)+(k+1)
    return img

def create_img_single(ins=[],isFromFile=True):
    hi,wi,ci,ki,ti,pi = ins[0],ins[1],ins[2],ins[3],ins[4],ins[5]
    ic(ci)
    if isFromFile is True:
        img = cv2.resize(cv2.imread("cute.jpg"),(hi,wi))
        temp = [list(get_channel_first(img)[0])]

        while(1):
            if len(temp)>=ci:
                break
            temp.append(np.ones([hi,wi]))
        #ic(np.array(temp).shape)
        img = get_channel_last(np.array(temp))
    else:
        img = np.zeros([hi,wi,ci])
        for i in range(hi):
            for j in range(wi):
                for k in range(ci):
                    img[i,j,k] = 100*(i+1)+10*(j+1)+(k+1)
    for i in range(hi):
        for j in range(wi):
            for k in range(1,ci):
                img[i,j,k]=0
    return img

def create_img_identical(ins=[],isFromFile=True):
    hi,wi,ci,ki,ti,pi = [ins[k] for k in ins.keys()]
    if isFromFile is True:
        img = cv2.resize(cv2.imread("cute.jpg"),(hi,wi))
        temp = list(get_channel_first(img))
        while(1):
            if len(temp)>=ci:
                break
            temp.append(np.ones([hi,wi]))
        #ic(np.array(temp).shape)
        img = get_channel_last(np.array(temp))
    else:
        img = np.zeros([hi,wi,ci])
        for i in range(hi):
            for j in range(wi):
                for k in range(ci):
                    img[i,j,k] = 100*(i+1)+10*(j+1)+(k+1)
    for i in range(hi):
        for j in range(wi):
            for k in range(1,ci):
                img[i,j,k]=img[i,j,0]
    return img
    
#################################
#  Dimensions , ins, outs,pi,po #
#################################
def get_dims(hi,wi,ci,ki,ti,ho,wo,co,ko,to,nslots=2**15):
    pi = get_p(nslots,hi,wi,ki,ti)
    po = get_p(nslots,ho,wo,ko,to)
    # ins =  [hi,wi,ci,ki,ti,pi]
    # outs = [ho,wo,co,ko,to,po]
    ho = int(hi/(ko/ki))
    wo = int(wi/(ko/ki))
    ins = {'h':hi, 'w':wi, 'c':ci, 'k':ki, 't':ti, 'p':pi}
    outs = {'h':ho, 'w':wo, 'c':co, 'k':ko, 't':to, 'p':po}
    return pi,ins,po,outs

def get_dims_p_assigned(hi,wi,ci,ki,ti,pi,ho,wo,co,ko,to,po,nslots=2**15):
    #pi = get_p(nslots,hi,wi,ki,ti)
    #po = get_p(nslots,ho,wo,ko,to)
    ins =  [hi,wi,ci,ki,ti,pi]
    outs = [ho,wo,co,ko,to,po]
    return ins,outs

        
def get_conv_params(conv_layer, ki, hi, wi):
    U = get_channel_last(np.array(conv_layer.weight.detach()))
    co, ci, fh, fw = conv_layer.weight.shape
    stride, stride = conv_layer.stride

    if stride == 1:
        ko = ki
        ho = hi
        wo = wi    
    else:
        ko = ki*stride
        ho = int(hi/stride)
        wo = int(wi/stride)

    ti = get_t(ci,ki)
    to = get_t(co,ko)
    pi,ins,po,outs = get_dims(hi,wi,ci,ki,ti,ho,wo,co,ko,to)
    return U, ins, outs

def prepare_bn2d(bn_layer, ki, hi, wi):
    return U, ins, outs

def prepare_fc(bn_layer, ki, hi, wi):
    return U, ins, outs

def forward_conv(layer, ctx, ki, hi, wi):
    U, ins, outs = get_conv_params(layer, ki=ki, hi=hi, wi=wi)    
    out, out2 = MultConv(ctx, U, ins, outs)
    un = unpack(out,outs)
    return out, out2, un