"""Analyze saved original-library GPU output against exact binary-input GEMM.
Usage: python3 analyze_gpu.py INPUT.bin OUTPUT.bin
"""
from pathlib import Path
import sys,struct,json
import numpy as np
from compare import exact_dot,metrics

def main():
    source=Path(sys.argv[1]).read_bytes(); output=Path(sys.argv[2]).read_bytes()
    m,n,k=struct.unpack('<III',source[:12])
    if output[:12]!=source[:12]: raise ValueError('shape/header mismatch')
    if len(source)!=12+8*(m*k+k*n) or len(output)!=12+16*m*n: raise ValueError('file length mismatch')
    data=np.frombuffer(source,dtype='<f8',offset=12)
    a=data[:m*k].reshape((m,k),order='F'); b=data[m*k:].reshape((k,n),order='F')
    data=np.frombuffer(output,dtype='<f8',offset=12)
    ref=data[:m*n].reshape((m,n),order='F'); candidate=data[m*n:].reshape((m,n),order='F')
    exact=[[exact_dot(a[i],b[:,j]) for j in range(n)] for i in range(m)]
    results={}
    for name,arr in [('cublas_dgemm',ref),('ozblas_D_H',candidate)]:
        results[name]=metrics(arr,exact) if np.all(np.isfinite(arr)) else {'status':'nonfinite_output','count':int(np.count_nonzero(~np.isfinite(arr)))}
    print(json.dumps(results,indent=2))

if __name__=='__main__': main()
