from muxcnn.utils import *

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



def get_dims_p_assigned(hi,wi,ci,ki,ti,pi,ho,wo,co,ko,to,po):
    ins =  [hi,wi,ci,ki,ti,pi]
    outs = [ho,wo,co,ko,to,po]
    return ins,outs


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





def __torch_forward(model, img_tensor, do_activation=False):
    res1_ = model.conv1(img_tensor)
    res1 = model.bn1(res1_)
    print("after BN", res1.min(), res1.max())
    if do_activation: res1 = model.activation(res1)
    

    ### basicblock1 (no striding)
    res2_ = model.layer1[0].conv1(res1)
    res2 = model.layer1[0].bn1(res2_)
    print("after BN", res2.min(), res2.max())
    if do_activation: res2 = model.activation(res2)

    res3_ = model.layer1[0].conv2(res2)
    res3 = model.layer1[0].bn2(res3_)
    res3 += res1 # simply add
    print("after BN", res3.min(), res3.max())
    if do_activation: res3 = model.activation(res3)

    #### basicblock2
    res4_ = model.layer2[0].conv1(res3)
    res4 = model.layer2[0].bn1(res4_)
    print("after BN", res4.min(), res4.max())
    if do_activation: res4 = model.activation(res4)

    res5_ = model.layer2[0].conv2(res4)
    res5 = model.layer2[0].bn2(res5_)

    # basicblock2 - shortcut
    sc1_conv1, sc1_bn1 = model.layer2[0].shortcut
    sc1 = sc1_bn1(sc1_conv1(res3))
    res5 += sc1
    print("after BN", res5.min(), res5.max())
    if do_activation: res5 = model.activation(res5)

    #### basicblock3
    res6_ = model.layer3[0].conv1(res5)
    res6 = model.layer3[0].bn1(res6_)
    print("after BN", res6.min(), res6.max())
    if do_activation: res6 = model.activation(res6)

    res7_ = model.layer3[0].conv2(res6)
    res7 = model.layer3[0].bn2(res7_)

    # basicblock3 - shortcut
    sc2_conv1, sc2_bn1 = model.layer3[0].shortcut
    sc2 = sc2_bn1(sc2_conv1(res5))
    res7 += sc2
    print("after BN", res7.min(), res7.max())
    if do_activation: res7 = model.activation(res7)

    # terminal
    res_avg = model.avgpool(res7)
    out = torch.flatten(res_avg, 1)
    out = model.linear(out)
    return out.detach().numpy(), F.log_softmax(out, dim=1).detach().numpy()
    