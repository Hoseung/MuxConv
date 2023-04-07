import numpy as np
from hemul import loader 
he = loader.load()
from hemul.comparator import _appr_sign_funs

# degrees from i=1 to k
# from Eunsang Lee+21
MINIMUM_MULT = {4:[3,3,5],
                5:[5,5,5],
                6:[3,5,5,5],
                7:[3,3,5,5,5],
                8:[5,5,5,5,9],
                9:[5,5,5,5,5,5],
               10:[5,5,5,5,5,9],
               11:[3,5,5,5,5,5,5],
               12:[3,5,5,5,5,5,9],
               13:[3,5,5,5,5,5,5,5],
               14:[3,3,5,5,5,5,5,5,5],
               15:[3,3,5,5,5,5,5,5,9],
               16:[3,3,5,5,5,5,5,5,5,5],
               17:[5,5,5,5,5,5,5,5,5,5],
               18:[3,3,5,5,5,5,5,5,5,5,5],
               19:[5,5,5,5,5,5,5,5,5,5,5],
               20:[5,5,5,5,5,5,5,5,5,5,9]}

MINIMUM_DEPTH = {4:[27],
                 5:[7,13],
                 6:[15,15],
                 7:[7,7,13],
                 8:[7,15,15],
                 9:[7,7,7,13],
                10:[7,7,13,15],
                11:[7,15,15,15],
                12:[15,15,15,15],
                13:[15,15,15,31],
                14:[7,7,15,15,27],
                15:[7,15,15,15,27],
                16:[15,15,15,15,27],
                17:[15,15,15,29,29],
                18:[15,15,29,29,31],
                19:[15,29,31,31,31],
                20:[29,31,31,31,31]}


class ApprSign_FHE():
    def __init__(self, 
                 hec,
                alpha=12, 
                margin = 0.03, 
                eps=0.01, 
                xmin=-1,
                xmax=1,
                min_depth=True, 
                min_mult=False,
                debug=False):
        self.hec = hec
        self.alpha = alpha
        self.margin = margin
        self.eps = eps
        self.xmin = xmin
        self.xmax = xmax
        self.min_depth = min_depth
        self.min_mult = ~min_depth
        self.funs = None
        self.degrees = None
        self.debug=debug
        if self.alpha is not None:
            self._set_degree()
        if self._params_set():
            self._set_funs()

    def _params_set(self):
        return self.degrees is not None and self.margin is not None and self.eps is not None

    def _set_degree(self):
        if self.min_depth:
            self.degrees = MINIMUM_DEPTH[self.alpha]
        elif self.min_mult:
            self.degrees = MINIMUM_MULT[self.alpha]
    
    def _set_funs(self, degrees=None, xmin=None, xmax=None):
        if degrees is None:
            degrees = self.degrees
        if xmin is None:
            xmin = self.xmin
        if xmax is None:
            xmax = self.xmax
        
        self.funs = _appr_sign_funs(degrees, xmin, xmax, 
                margin=self.margin, eps=self.eps)
        if self.debug: print("function approximators set")
        if self.debug: print(f"degrees = {self.degrees}, margin = {self.margin}, eps = {self.eps}") 

    def __call__(self, xin):
        if self.funs is not None:
            for fun, deg in self.funs:
                if self.debug: 
                    print("APPR", deg, xin.logp, xin.logq)
                    tmp = self.hec.decrypt(xin)
                    print(tmp, flush=True)
                    print("min max", tmp.min(), tmp.max(), flush=True)
                if xin.logq <= ((1+np.ceil(np.log2(deg))) * self.hec.parms.logp):
                    xin = self.hec.bootstrap2(xin)
                    if self.debug: print("AFTER bootstrap", xin.logp, xin.logq)

                xin = self.hec.function_poly(fun.coef, xin)#, , xin.logp)
            
            if xin.logq <= (3*self.hec.parms.logp):
                xin = self.hec.bootstrap2(xin)
            return xin
        else:
            self._set_funs()
            return self.__call__(xin)

class ApprRelu_HEAAN(ApprSign_FHE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, xin):
        hec = self.hec
        if xin.logq <= (3 * self.hec.parms.logp):
            xin = self.hec.bootstrap2(xin)
        out = ApprSign_FHE.__call__(self, he.Ciphertext(xin))
        
        if self.debug: 
            print("After ApprSign", out.logp, out.logq)
            tmp = self.hec.decrypt(out)
            print(tmp, flush=True)
            print("min max", tmp.min(), tmp.max(), flush=True)
        
        tmp = hec.addConst(out, np.repeat(1, hec.parms.n), inplace=False)
        
        if self.debug: 
            print("After +1", tmp.logp, tmp.logq)
            zzz = self.hec.decrypt(tmp)
            print(zzz, flush=True)
            print("min max", zzz.min(), zzz.max(), flush=True)

        tmp = hec.multByVec(tmp, np.repeat(1/2, hec.parms.n), inplace=False)
        
        if self.debug: 
            print("After *1", tmp.logp, tmp.logq)
            zzz = self.hec.decrypt(tmp)
            print(zzz, flush=True)
            print("min max", zzz.min(), zzz.max(), flush=True)

        hec.rescale(tmp)
        
        if self.debug: 
            print(xin, tmp)
        if xin.logq > tmp.logq:
            hec.match_mod(xin, tmp)
        elif xin.logq < tmp.logq:
            hec.match_mod(tmp, xin)
        hec.mult(xin, tmp, inplace=True)

        if self.debug: 
            print("After mult", xin.logp, xin.logq)
            zzz = self.hec.decrypt(xin)
            print(zzz, flush=True)
            print("min max", zzz.min(), zzz.max(), flush=True)

        hec.rescale(xin)
        return xin