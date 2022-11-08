import torch
import numpy as np
from matplotlib import pyplot as plt
from math import ceil
from PIL import Image
import torchvision.transforms as transforms

### Auxillary functions for the MUXCNN
def load_params(model, fn_param, device):
    trained_param = torch.load(fn_param, map_location = torch.device(device))
    trained_param = {key : value.cpu()   for key,value in trained_param.items()}
    model.load_state_dict(trained_param)

def load_img(fname, hi=None, wi=None):
    image = Image.open(fname)
    transform = transforms.Compose([
        transforms.Resize((hi,wi)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    img = transform(image).unsqueeze(0)
    return img.type(torch.FloatTensor)

def compare(tout, fout, ch=0, fn=None, err_mag= 1e-5):
    tt = tout.detach().numpy()[0,ch,:,:]
    ff = fout[ch,:,:]
    
    fig, axs = plt.subplots(2,2)
    vmin = tt.min()
    vmax = tt.max()
    axs[0,0].imshow(tt, vmin=vmin, vmax=vmax)
    axs[0,0].set_title("torch")
    axs[0,1].imshow(ff, vmin=vmin, vmax=vmax)
    axs[0,1].set_title("fhe")
    axs[1,0].imshow(tt-ff, vmin=vmin*err_mag, vmax=vmax*err_mag)
    axs[1,0].set_title(f"diff (min,max)*{err_mag:.2g}")
    plt.tight_layout()
    if fn is not None:
        plt.savefig(fn.replace(".png", f"_{ch}.png"))
        plt.close()

def print_compare(tt, ff):
    for ch in [0,1,8,15]:
        print(tt[0,ch,4:6,4:6])
        print(ff[ch,4:6,4:6], "--\n")

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
#################  auxillary  #################



def SumSlots(ct_a,m,p):
    nrots = 0
    n = int(np.floor(np.log2(m)))
    ct_b = []
    ct_b.append(ct_a)
    for j in range(1,n+1):
        lrots = int(-p*2**(j-1))
        ct_b.append(ct_b[j-1]+np.roll(ct_b[j-1],lrots))
        if lrots!=0:
            nrots=nrots+1  #______________________________ROTATION
    ct_c = ct_b[n]
    for j in range(0,n):
        n1 = np.floor((m/(2**j))%2)
        if n1==1:
            n2 =int(np.floor((m/(2**(j+1)))%2))
            lrots = int(-p*2**(j+1))*n2
            ct_c += np.roll(ct_b[j],lrots)
            if lrots!=0:
                nrots=nrots+1#____________________________ROTATION

    return ct_c,nrots


#################################
#  Dimensions , ins, outs,pi,po #
#################################
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
        print("get_channel_first called")
        return out
    elif len(dim)==3:
        return U.transpose(2,0,1)

def get_channel_last(U):
    dim=U.shape
    if len(dim)==4:
        return U.transpose(2,3,1,0)
    elif len(dim)==3:
        return U.transpose(1,2,0)
    
def get_dims(hi,wi,ci,ki,ti,ho,wo,co,ko,to,nslots=2**15):
    pi = get_p(nslots,hi,wi,ki,ti)
    po = get_p(nslots,ho,wo,ko,to)
    ho = int(hi/(ko/ki))
    wo = int(wi/(ko/ki))
    ins = {'h':hi, 'w':wi, 'c':ci, 'k':ki, 't':ti, 'p':pi}
    outs = {'h':ho, 'w':wo, 'c':co, 'k':ko, 't':to, 'p':po}
    return pi,ins,po,outs


def get_conv_params(conv_layer, ins):
    U = get_channel_last(conv_layer.weight.detach().numpy())
    co, ci, fh, fw = conv_layer.weight.shape
    stride, stride = conv_layer.stride

    ki, hi, wi = ins['k'], ins['h'], ins['w']

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

def plot_4x4(img, co=None, fn=None):
    fig, axs = plt.subplots(4,4, figsize=(8,8))
    axs = axs.ravel()
    vmin = img.min()
    vmax = img.max()
    if co is None:
        co = len(img)
    for ico in range(co):
        axs[ico].imshow(img[ico], vmin=vmin, vmax=vmax)
    plt.tight_layout()
    if fn is not None:
        plt.savefig(fn)
        plt.close()
