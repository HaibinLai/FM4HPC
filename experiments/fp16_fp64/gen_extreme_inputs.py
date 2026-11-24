import numpy as np, struct, sys, json
out=sys.argv[1]
def U(r,s): return r.uniform(-1,1,s)
def mk(name,M,N,K,seed):
    r=np.random.default_rng(seed)
    a=U(r,(M,K)); b=U(r,(K,N))
    if name=='spread_100':  a=np.ldexp(a,r.integers(-100,101,a.shape)); b=np.ldexp(b,r.integers(-100,101,b.shape))
    elif name=='spread_300':a=np.ldexp(a,r.integers(-300,301,a.shape)); b=np.ldexp(b,r.integers(-300,301,b.shape))
    elif name=='big_400':   a=np.ldexp(a,400); b=np.ldexp(b,400)          # products ~2^800
    elif name=='overflow':  a=np.ldexp(a,600); b=np.ldexp(b,600)          # products ~2^1200 -> inf
    elif name=='tiny_500':  a=np.ldexp(a,-500); b=np.ldexp(b,-500)        # products ~2^-1000
    elif name=='underflow': a=np.ldexp(a,-560); b=np.ldexp(b,-560)        # products ~2^-1120 subnormal/0
    elif name=='subnormal': a=np.ldexp(a,-1060); b=np.ldexp(b,0)          # A entries subnormal
    elif name=='row_scale': a=np.ldexp(a,(r.integers(-200,201,(M,1)))); b=np.ldexp(b,(r.integers(-200,201,(1,N))))
    elif name=='outlier':   a[0,0]=2.0**500; b[0,0]=2.0**-3
    return a,b
names=['spread_100','spread_300','big_400','overflow','tiny_500','underflow','subnormal','row_scale','outlier']
n=0
for nm in names:
    for (M,N,K) in [(16,16,16),(3,3,16)]:
        a,b=mk(nm,M,N,K,7000+n); n+=1
        with open(f'{out}/e_{nm}_{M}x{N}x{K}.bin','wb') as f:
            f.write(struct.pack('<III',M,N,K)); f.write(a.astype('<f8').tobytes(order='F')); f.write(b.astype('<f8').tobytes(order='F'))
print(n, 'inputs; min|a|',float(np.min(np.abs(a[a!=0]))))
