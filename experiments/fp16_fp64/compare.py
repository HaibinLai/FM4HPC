"""OzBLAS D_H arithmetic experiment, not execution of the CUDA library.
Models double/half splitting, FP32 partial dots, and sumMode=2 FP64 summation.
Completed residual rows are explicitly zero-padded (an extension of source).
Alternative summation orders are experiments, not upstream implementations.
"""
from pathlib import Path
from fractions import Fraction as F
import json, math, platform, struct
import numpy as np

ROOT=Path(__file__).resolve().parent
COMMIT='a6f6198bc7af6cf288607ef867693f991cf48862'

def exact_dot(x,y):
    return sum((F(float(a))*F(float(b)) for a,b in zip(x,y)),F(0))

def split_rows(a,limit):
    r=a.copy(); k=a.shape[1]
    rho=math.ceil(53-(24-math.log2(max(2,k)))/2)
    ds=[]; es=[]; frozen=0
    for s in range(limit):
        if not np.any(r): break
        d=np.zeros(r.shape,dtype=np.float16); ex=np.zeros(r.shape[0],dtype=np.int32)
        for i in range(r.shape[0]):
            mx=float(np.max(np.abs(r[i])))
            if mx==0:
                frozen+=1; continue  # source calls log2(0); explicit model extension
            tau=math.ceil(math.log2(mx)); ex[i]=tau
            sigma=math.ldexp(.75,rho+tau)
            if not math.isfinite(sigma): raise ValueError('sigma overflow')
            piece=(r[i]+sigma)-sigma
            residual=r[i]-piece
            scaled=np.ldexp(piece,-tau)
            d[i]=scaled
            if not np.array_equal(d[i].astype(np.float64),scaled):
                raise ValueError('slice not exactly representable in FP16')
            for x,h,t in zip(r[i],piece,residual):
                assert F(float(x))==F(float(h))+F(float(t)), 'non-exact residual subtraction'
            r[i]=residual
        ds.append(d); es.append(ex)
    return ds,es,r,{'rho':rho,'slices':len(ds),'zero_padded_row_steps':frozen,
                   'max_residual':float(np.max(np.abs(r)))}

def reduction(values,dtype,order):
    values=[dtype(v) for v in values]
    if order=='reverse': values=values[::-1]
    if order=='pairwise':
        while len(values)>1:
            values=[dtype(values[i]+values[i+1]) if i+1<len(values) else values[i]
                    for i in range(0,len(values),2)]
        return dtype(values[0]) if values else dtype(0)
    out=dtype(0)
    for v in values: out=dtype(out+v)
    return out

def partial(x,y):
    products=[np.float32(a)*np.float32(b) for a,b in zip(x,y)]
    target=exact_dot(x,y)
    out=[reduction(products,np.float32,o) for o in ('forward','reverse','pairwise')]
    if not all(F(float(v))==target for v in out):
        raise AssertionError('tested FP32 partial sum lost exactness')
    return float(out[0])

def emulate(a,b,limit):
    da,ea,ra,sa=split_rows(a,limit); db,eb,rb,sb=split_rows(b.T,limit)
    terms=[]
    for level in range(len(da)+len(db)-1):
        for i in range(len(da)):
            j=level-i
            if not 0<=j<len(db): continue
            term=np.empty((a.shape[0],b.shape[1]))
            for x in range(a.shape[0]):
                for y in range(b.shape[1]):
                    term[x,y]=math.ldexp(partial(da[i][x],db[j][y]),int(ea[i][x]+eb[j][y]))
            terms.append(term)
    outputs={}
    for order in ('forward','reverse','pairwise'):
        outputs[order]=np.array([[reduction([t[x,y] for t in terms],np.float64,order)
            for y in range(b.shape[1])] for x in range(a.shape[0])])
    # Accurate host summation is a control, NOT upstream NearSum (unsupported for D_H).
    outputs['math_fsum']=np.array([[math.fsum(t[x,y] for t in terms)
        for y in range(b.shape[1])] for x in range(a.shape[0])])
    return outputs,{'A':sa,'B':sb,'partial_gemms':len(terms)}

def metrics(out,ref):
    errors=[abs(F(float(out[i,j]))-ref[i][j]) for i in range(out.shape[0]) for j in range(out.shape[1])]
    scale=max(abs(v) for row in ref for v in row)
    rounded=np.array([[float(v) for v in row] for row in ref])
    return {'max_abs':float(max(errors)),'normalized_max':float(max(errors)/scale) if scale else None,
            'different_from_rounded_exact':int(np.count_nonzero(out!=rounded))}

def save_input(name,a,b):
    # Header: uint32 M,N,K little-endian; then column-major binary64 A,B.
    p=ROOT/(name+'.bin')
    with p.open('wb') as f:
        f.write(struct.pack('<III',a.shape[0],b.shape[1],a.shape[1]))
        f.write(a.astype('<f8').tobytes(order='F')); f.write(b.astype('<f8').tobytes(order='F'))

def main():
    rng=np.random.default_rng(20261122)
    cases=[('uniform_k16',rng.uniform(-1,1,(3,16)),rng.uniform(-1,1,(16,3))),
           ('uniform_k64',rng.uniform(-1,1,(3,64)),rng.uniform(-1,1,(64,3))),
           ('spread_40',np.ldexp(rng.uniform(-1,1,(3,16)),rng.integers(-20,21,(3,16))),
             np.ldexp(rng.uniform(-1,1,(16,3)),rng.integers(-20,21,(16,3)))),
           ('cancellation',np.tile([1.,2.**-45,-1.,2.**-45],(3,1)),np.ones((4,3)))]
    rows=[]; witnesses=[]
    for name,a,b in cases:
        save_input(name,a,b)
        ref=[[exact_dot(a[i],b[:,j]) for j in range(b.shape[1])] for i in range(a.shape[0])]
        rows.append({'case':name,'method':'numpy_fp64',**metrics(a@b,ref)})
        # Naive baseline only when direct conversion remains finite.
        with np.errstate(over='ignore',invalid='ignore'):
            ah=a.astype(np.float16).astype(np.float32); bh=b.astype(np.float16).astype(np.float32)
        if np.all(np.isfinite(ah)) and np.all(np.isfinite(bh)):
            out=np.array([[reduction([np.float32(x*y) for x,y in zip(ah[i],bh[:,j])],np.float32,'forward')
               for j in range(b.shape[1])] for i in range(a.shape[0])],dtype=np.float64)
            rows.append({'case':name,'method':'naive_fp16_fp32',**metrics(out,ref)})
        else: rows.append({'case':name,'method':'naive_fp16_fp32','status':'conversion_overflow'})
        for limit in (3,6,12):
            outs,meta=emulate(a,b,limit)
            for order,out in outs.items():
                rows.append({'case':name,'method':f'ozblas_model_s{limit}_{order}',**meta,**metrics(out,ref)})
            diff=outs['forward']!=outs['reverse']
            if np.any(diff):
                i,j=np.argwhere(diff)[0]
                witnesses.append({'case':name,'limit':limit,'index':[int(i),int(j)],
                    'forward_hex':float(outs['forward'][i,j]).hex(),
                    'reverse_hex':float(outs['reverse'][i,j]).hex(),
                    'exact_fraction':str(ref[i][j]),
                    'max_order_difference':float(np.max(np.abs(outs['forward']-outs['reverse'])))})
    result={'scope':'CPU arithmetic model + explicit zero-row extension; not GPU/library execution',
        'upstream_commit':COMMIT,'seed':20261122,'numpy':np.__version__,'python':platform.python_version(),
        'configuration':{'fastMode':0,'sumMode':2,'splitEpsMode':0,'splitShift':1,'alpha':1,'beta':0},
        'rows':rows,'summation_order_witnesses':witnesses}
    (ROOT/'results.json').write_text(json.dumps(result,indent=2))
    for row in rows:
        if row['method'] in ('numpy_fp64','naive_fp16_fp32','ozblas_model_s3_forward','ozblas_model_s6_forward','ozblas_model_s12_forward'):
            print(row['case'],row['method'],row.get('normalized_max',row.get('status')))
    print('order witnesses:',len(witnesses))

if __name__=='__main__': main()
