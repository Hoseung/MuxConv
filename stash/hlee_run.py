from hlee_cnn import *
from matplotlib import pyplot as plt
from hlee_utils import *
from hlee_multiplexed_lee22e import Multiplexed_lee22e as multiplexed


class CompareConv:
    def __init__(self,fimage,fparam,kernel_size ,co):
        self.fimage = fimage#"./cute.jpg"
        self.fparam = fparam#"./models/simple_model_hlee.pt"
        self.device = 'cpu'
        self.kernel_size = kernel_size
        self.nslots =2**15
        


        self.hi,self.wi,self.ci,self.ki,self.ti=32,32,3, 1,3
        self.ho,self.wo,self.co,self.ko,self.to=32,32,co,1,4




        #***************************
        self.fh,self.fw = kernel_size,kernel_size
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog',      'frog',       'horse','ship','truck']
        self.U_multconv=None
        self.U_torchconv=None
        self.torout = None
        self.mpout=None
        self.unpacked_mpout=None
        self.tot_rots=0
        self.run_torchconv()
        self.run_multconv()

    def run_torchconv(self): 
        print("run torchconv")
        print(self.co)
        tor = TorchConv_infer( num_classes=len(self.classes),
                               activation=F.relu,
                               fn_param=self.fparam,
                               device=self.device,
                               wo=self.wo,
                               ho=self.ho,
                               co=self.co)
        self.torout = tor.TorchConv(self.fimage)
        self.U_torchconv = tor.U_torchconv #with channel last.
        self.U_multconv  = tor.U_multconv #with channel last.


    def run_multconv(self):
        mp = multiplexed([self.hi,self.wi,self.ci,self.ki,self.ti],
                         [self.ho,self.wo,self.co,self.ko,self.to],
                         [self.fh,self.fw],
                         self.nslots,
                         self.device,self.classes)
        print(f"U_mpconv : {self.U_multconv.shape}")
        img = cv2.imread(self.fimage)
        img = cv2.resize(img,(self.hi,self.wi))
        mpout,self.tot_rots = mp.MultConv(mp.MultPack(img,self.ki,self.ti),self.U_multconv)
        self.mpout = mpout
        self.unpacked_mpout = mp.unpack(self.mpout,self.ho,self.wo,self.co,self.ko,self.to)
    
    def compare(self):
        return self.torout,self.mpout,self.unpacked_mpout,self.tot_rots


    def unpack(self,A,ha,wa,ca,ka,ta):
        mp = multiplexed([self.hi,self.wi,self.ci,self.ki,self.ti],
                         [self.ho,self.wo,self.co,self.ko,self.to],
                         [self.fh,self.fw],
                         self.nslots,
                         self.device,self.classes)
        return mp.unpack(A,ha,wa,ca,ka,ta)
    
        
