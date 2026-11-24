"""Source-guided ozIMMU arithmetic model; NOT execution/verification of CUDA.
Python >=3.9 + NumPy. Run from any directory; results go beside this script.
Scope: real NN GEMM, alpha=1, beta=0, finite NORMAL nonzero inputs,
exponent shift <128, finite intermediate scaling. GPU lowering is unverified.
"""
from fractions import Fraction as F
from pathlib import Path
import contextlib, io, json, math, platform, struct
import numpy as np

COMMIT = '08eea9231729d54dbfd92955f2cbfc21ec236856'
MASK128 = (1 << 128) - 1

def bits(x):
    return struct.unpack('<Q', struct.pack('<d', float(x)))[0]

def p2(e):
    return F(2**e) if e >= 0 else F(1, 2**(-e))

def width(k):
    if not 1 <= k < 2**30:
        raise ValueError('k outside modeled get_bits_per_int8 domain')
    return min(7, (31 - (k-1).bit_length()) // 2)

def split_rows(a, slices, p):
    digits = np.zeros((slices, *a.shape), dtype=np.int64)
    scales = []
    for row, values in enumerate(a):
        eb = [(bits(x) >> 52) & 2047 for x in values]
        if any(e in (0, 2047) for e in eb):
            raise ValueError('zero/subnormal/NaN/Inf outside source-faithful domain')
        max_e = max(eb) + 1
        if max_e >= 2047:
            raise ValueError('source x2(max_exp) overflows')
        scale = math.ldexp(1.0, max_e-1023)
        scales.append(scale)
        for col, x in enumerate(values):
            offset = max_e-eb[col]
            if offset >= 128:
                raise ValueError('source uint128 shift >=128 has unestablished semantics')
            mantissa = ((bits(x) & ((1 << 52)-1)) | (1 << 52)) << 75
            shifted = mantissa >> offset
            for s in range(slices):
                d = (shifted >> (128-p)) * (1 if x > 0 else -1)
                digits[s,row,col] = d
                shifted = (shifted << p) & MASK128
                # Independent exact rational digit specification (not FP rounding).
                q = abs(F(float(x))) / (2*F(scale)) * p2(p*(s+1))
                expected = (q.numerator // q.denominator) % (1 << p)
                assert abs(d) == expected
    return digits, np.array(scales)

def model(a, b, slices):
    p = width(a.shape[1])
    aa, sa = split_rows(a,slices,p)
    bb, sb = split_rows(b.T,slices,p)
    acc = np.zeros((a.shape[0],b.shape[1]),dtype=np.float64)
    max_dot = 0
    # config.cu: ascending diagonal sum, then ascending A_id.
    for diagonal in range(slices):
        for i in range(diagonal+1):
            j = diagonal-i
            dot = aa[i] @ bb[j].T  # int64 prevents silent host INT32 overflow
            max_dot = max(max_dot,int(np.max(np.abs(dot))))
            assert np.all(np.abs(dot) <= 2**31-1)
            shift = p*(i+j)-2*(7-p)
            # gemm.cu: signed INT64 shift represented mathematically; target
            # compiler semantics for negative left shifts remain a trust boundary.
            term = np.ldexp(dot.astype(np.float64),32) * math.ldexp(1.0,-shift)
            acc = acc + term
    out = ((acc / 2**44) * sa[:,None]) * sb[None,:]
    if not np.all(np.isfinite(out)):
        raise ValueError('nonfinite reconstructed output')
    return out, p, max_dot

def oracle(a,b):
    return [[sum((F(float(a[i,k]))*F(float(b[k,j])) for k in range(a.shape[1])),F(0))
             for j in range(b.shape[1])] for i in range(a.shape[0])]

def metrics(x, exact):
    errs = [abs(F(float(x[i,j]))-exact[i][j]) for i in range(x.shape[0]) for j in range(x.shape[1])]
    scale = max(abs(v) for row in exact for v in row)
    maxerr = max(errs)
    rounded = np.array([[float(v) for v in row] for row in exact])
    return {'max_abs_error':float(maxerr),'max_error_over_max_exact':float(maxerr/scale) if scale else None,
            'different_from_correctly_rounded':int(np.count_nonzero(x != rounded)),
            'elements':x.size}

def main():
    rng=np.random.default_rng(20261122)
    cases=[]
    for k in (3,16,64,256):
        cases.append((f'uniform_k{k}',rng.uniform(-1,1,(4,k)),rng.uniform(-1,1,(k,4))))
    cases.append(('exponent_spread_80',np.ldexp(rng.uniform(1,2,(4,32)),rng.integers(-40,41,(4,32))),
                  np.ldexp(rng.uniform(1,2,(32,4)),rng.integers(-40,41,(32,4)))))
    cases.append(('cancellation',np.array([[1.,2.**-45,-1.,2.**-45]]),np.ones((4,1))))
    cases.append(('positive_integers',np.array([[1.,2.,3.,4.]]),np.array([[2.],[3.],[4.],[5.]])))
    rows=[]
    for name,a,b in cases:
        ex=oracle(a,b); standard=a@b
        rows.append({'case':name,'method':'numpy_fp64','shape':[a.shape[0],a.shape[1],b.shape[1]],**metrics(standard,ex)})
        for s in (3,6,9,12,18):
            out,p,md=model(a,b,s)
            rows.append({'case':name,'method':f'ozimmu_cpu_s{s}','bits_per_slice':p,'max_int_dot':md,
                         'different_from_numpy':int(np.count_nonzero(out != standard)),**metrics(out,ex)})
    rejected=[]
    for name,a in [('zero',np.array([[0.,1.]])),('subnormal',np.array([[np.nextafter(0.,1.),1.]])),
                   ('shift_128',np.array([[1.,2.**-127]])),('max_double',np.array([[np.finfo(float).max,1.]]))]:
        try: model(a,np.ones((2,1)),9)
        except ValueError as e: rejected.append({'case':name,'reason':str(e)})
        else: raise AssertionError('guard failed: '+name)
    # Exact integer safety checks at width transition points, not a universal proof.
    bounds=[]
    for k in (1,3,2**17,2**17+1,2**19,2**19+1,2**29):
        p=width(k); bound=k*((1<<p)-1)**2
        assert bound < 2**31
        bounds.append({'k':k,'p':p,'absolute_sum_bound':bound})
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream): np.show_config()
    result={'scope':'CPU source-guided arithmetic model, NOT CUDA execution or Lean proof',
      'source_commit':COMMIT,'seed':20261122,'python':platform.python_version(),'platform':platform.platform(),
      'numpy':np.__version__,'numpy_config':stream.getvalue(),'rows':rows,'rejected_domains':rejected,'integer_bound_checks':bounds,
      'cancellation_inputs':{'A':[1.,2.**-45,-1.,2.**-45],'B':[1.,1.,1.,1.],'exact':'1/17592186044416'}}
    path=Path(__file__).with_name('results.json'); path.write_text(json.dumps(result,indent=2))
    for r in rows:
        print(r['case'],r['method'],f"abs={r['max_abs_error']:.3e}",f"normalized={r['max_error_over_max_exact']:.3e}")
    print('Results:',path)

if __name__=='__main__': main()
