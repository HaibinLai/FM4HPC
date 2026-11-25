import sys
p = sys.argv[1]; s = open(p).read()
s = s.replace('#include <math.h>', '', 1)
s = '#include <math.h>\n#include <stdlib.h>\n' + s
a = "screate_A_x_b(&A, nrhs, &b, &ldb, &xtrue, &ldx, fp, postfix, &grid);"
assert a in s
s = s.replace(a, a + '\n    { char *nr = getenv("NANROW"); if (nr) { b[atoi(nr)] = NAN; printf("PROBE injected NaN at b[%d]\\n", atoi(nr)); } }', 1)
c = "&LUstruct, &SOLVEstruct, err_bounds, &stat, &info, dxtrue);"
assert c in s
s = s.replace(c, c + '\n        { int nn = 0, m_loc = ((NRformat_loc *)A.Store)->m_loc; for (int i = 0; i < m_loc; ++i) nn += !isfinite(b[i]);'
              ' printf("PROBE psgssvx_d2: info=%d nonfinite(x)=%d/%d err_bounds: normwise=%g componentwise=%g berr=%g\\n",'
              ' info, nn, m_loc, err_bounds[0], err_bounds[nrhs], err_bounds[2*nrhs]); fflush(stdout); }', 1)
open(p, 'w').write(s)
