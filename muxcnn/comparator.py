import numpy as np
from numpy import polynomial as P

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

def signv(x):
    ret = np.ones(len(x))
    ret[x<0] = -1
    return ret

def minimax(xl1, xr1, xl2, xr2, deg, npoints = 100):
    xx = np.concatenate((np.linspace(xl1,xr1,npoints), 
                         np.linspace(xl2,xr2,npoints)))
    chev = P.chebyshev.Chebyshev.fit(xx, signv(xx), deg=deg) # F.elu
    power = chev.convert(kind=P.Polynomial)
    return power

def _appr_sign_funs(degrees, xmin = -1, xmax = 1,
                    margin = 0.0005, eps=0.02): 
    xin = np.linspace(xmin,#  + 0.1*margin,
                      xmax, 100000)

    funs=[]
    for deg in degrees:
        #print(deg, eps, xin.min()-margin, -eps+margin, eps-margin, xin.max()+margin)
        fun = minimax(xin.min()-margin, -eps+margin, eps-margin, xin.max()+margin, deg, npoints = 2*deg+1)
        #print("[HEMUL] EPS before", eps)

        diff_r = max(abs(fun(xin[xin >= eps]) - 1))
        diff_l = max(abs(fun(xin[xin <= -eps]) + 1))
        eps = 1 - max([diff_r, diff_l])
        #eps = min([eps, 0.9])
        #print("[HEMUL] EPS now", eps)

        xin = fun(xin)
        #eps = min([abs(fun(-eps+margin)), fun(eps-margin)])
        #eps = min([eps, 0.9])
        funs.append((fun, deg))
    return funs
                    
def appr_sign(xin, xmin=-1, xmax=1, alpha=10, margin = 0.01, eps=0.02, min_depth=True, min_mult=False):
    """approximate sign function
    
    parameters
    ----------
    ctxt: Ciphertext
    alpha: positive int, tolerance parameter. err <= 2**alpha
    """
    #xin = np.linspace(xmin, xmax, 100000)
    if min_depth:
        degrees = MINIMUM_DEPTH[alpha]
    elif min_mult:
        degrees = MINIMUM_MULT[alpha]

    funs = _appr_sign_funs(degrees, xmin, xmax, 
                margin=margin, eps=eps)
    for fun in funs:
        xin = fun(xin)
    return xin

def appr_relu(xin, xmin, xmax, *args, **kwargs):
    out = appr_sign(xin, xmin, xmax, *args, **kwargs)
    #xin = np.linspace(xmin, xmax, 100000)
    return xin * (out+1)/2


class ApprSign():
    def __init__(self, 
                alpha=12, 
                margin = 0.01, 
                eps=0.02, 
                xmin=-1,
                xmax=1,
                min_depth=True, 
                min_mult=False,
                debug=False):
        self.alpha = alpha
        self.margin = margin
        self.eps = eps
        self.xmin = xmin
        self.xmax = xmax
        self.min_depth = min_depth
        self.min_mult = min_mult
        self.funs = None
        self.degrees = None
        self.debug = debug
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
        print("functions set")
        print(f"degrees = {self.degrees}, margin = {self.margin}, eps = {self.eps}") 

    def __call__(self, xin):
        if self.funs is not None:
            for fun, deg in self.funs:
                xin = fun(xin)
                if self.debug: 
                    print(f"degree = {deg}")
                    print(f"min, max", xin.min(), xin.max())

            return xin
        else:
            self._set_funs()
            return self.__call__(xin)

class ApprRelu(ApprSign):
    """

    example

    >>> ev = Evaluator(context)
    >>> appr = ApprRelu_FHE(ev, xmin=-10, xmax=10, min_depth=True)
    >>> activated = appr(ctx_a)

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, xin):
        out = ApprSign.__call__(self, xin)
        return xin * (out+1)/2

        
