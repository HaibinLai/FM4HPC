# Functions that pass a local status variable (iinfo, ierr, linfo, info2, ...) by address to a sub-call and never read
# it afterwards: the sub-call's failure is invisible to the caller.  For each such variable we list the callees that
# write it.  Usage: python3 dropped_status_scan.py <magma-2.10.0 root> [glob ...]
import re, sys, glob, os
root = sys.argv[1]
pats = sys.argv[2:] or ['src/z*.cpp']
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'argnum_scan.py')).read(); ns = {}
exec(src.split('out = []')[0].replace("root = sys.argv[1]", "root=''"), ns)
VAR = r'(iinfo|ierr|linfo|info2|iinfo2|lapack_info|info_tmp|tmp_info|lwork_info|info_)'
for pat in pats:
    for f in sorted(glob.glob(os.path.join(root, pat))):
        s = ns['zonly'](ns['strip'](open(f, errors='ignore').read()))
        for m in re.finditer(r'\b(magma_\w+)\s*\(([^;{}()]*(?:\([^()]*\)[^;{}()]*)*)\)\s*\{', s):
            i = m.end(); d = 1
            while d and i < len(s): d += (s[i] == '{') - (s[i] == '}'); i += 1
            body = s[m.end():i]
            for v in sorted(set(re.findall(r'&\s*' + VAR + r'\b', body))):
                reads = re.findall(r'(?<![&\w])' + v + r'\b(?!\s*=[^=])', body)
                # occurrences that are not "&v" and not "v = ..." (assignment) and not declarations
                decl = len(re.findall(r'magma_int_t[^;]*\b' + v + r'\b', body))
                if len(reads) - decl <= 0:
                    calls = sorted(set(re.findall(r'(\w+)\s*\([^;]*?&\s*' + v + r'\b', body)))
                    line = s[:m.start()].count('\n') + 1
                    print('%s:%d %s  %s never read; written by: %s' % (os.path.relpath(f, root), line, m.group(1), v, ', '.join(calls)))
