"""Compile an unchanged upstream kernel body as a single-thread host replay.
This checks sequential arithmetic only, not the CUDA binary or launch behavior.
"""
from pathlib import Path
import json, subprocess, hashlib

root=Path(__file__).resolve().parents[1]
source=root/'operators/vendor/ozIMMU/src/gemm.cu'
text=source.read_text()
start=text.index('template <bool is_beta_zero>')
brace=text.index('{',start); depth=1; end=brace+1
while depth:
    depth+=(text[end]=='{')-(text[end]=='}'); end+=1
body=text[start:end]
prefix='''#include <cstdio>
#include <cstddef>
#define __global__
struct Dim { unsigned x; };
Dim blockIdx{0}, blockDim{1}, threadIdx{0};
struct cuDoubleComplex { double x,y; };
cuDoubleComplex make_cuDoubleComplex(double x,double y) { return {x,y}; }
'''
suffix='''
int main() {
  cuDoubleComplex c{1,2}, beta{3,4};
  init_c_complex_kernel<false>(1,1,&c,1,beta);
  std::printf("%.17g %.17g\\n",c.x,c.y);
}
'''
out=Path(__file__).parent
cpp=out/'complex_beta_replay.cpp'; cpp.write_text(prefix+body+suffix)
binary=out/'complex_beta_replay'
subprocess.run(['clang++','-std=c++17','-O0',str(cpp),'-o',str(binary)],check=True)
run=subprocess.run([str(binary)],check=True,text=True,capture_output=True)
actual=[float(x) for x in run.stdout.split()]
expected=complex(1,2)*complex(3,4)
assert actual==[-5.,-14.]
assert [expected.real,expected.imag]==[-5.,10.]
result={'scope':'exact upstream kernel body, host single-thread replay; NOT GPU execution',
        'source_commit':'08eea9231729d54dbfd92955f2cbfc21ec236856',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'compiler':subprocess.check_output(['clang++','--version'],text=True).splitlines()[0],
        'input_C':[1,2],'beta':[3,4],'source_result':actual,
        'python_complex_reference':[expected.real,expected.imag],
        'conclusion':'sequential overwrite violates complex beta*C for this example',
        'gpu_reproduction':'pending','upstream_issue_submitted':False}
(out/'complex_beta_result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
