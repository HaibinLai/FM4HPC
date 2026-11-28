# For each extern "C" function with a `magma_int_t *info` (or `* info`) parameter, report whether the body assigns
# `*info = 0` (or info[0] = 0) before the first read of *info (`*info != 0`, `*info == 0`, `*info < 0`, `if (*info)`).
import re, sys, os, glob
root = sys.argv[1]
files = sorted(glob.glob(root + '/src/z*.cpp') + glob.glob(root + '/magmablas/z*.cpp') + glob.glob(root + '/magmablas/z*.cu')
               + glob.glob(root + '/src/*.cpp'))
seen=set(); hits=[]
for f in files:
    if f in seen: continue
    seen.add(f)
    s = open(f, errors='ignore').read()
    s_nc = re.sub(r'/\*.*?\*/', lambda m: ' ' * len(m.group(0)), s, flags=re.S)
    s_nc = re.sub(r'//[^\n]*', lambda m: ' ' * len(m.group(0)), s_nc)
    for m in re.finditer(r'\n(?:extern "C"\s*)?(?:magma_int_t|void|int)\s*\n?\s*(magma_\w+)\s*\(([^;{]*?)\)\s*\{', s_nc):
        name, params = m.group(1), m.group(2)
        if not re.search(r'magma_int_t\s*\*\s*info\b', params): continue
        # body: match braces
        i = m.end(); depth = 1
        while depth and i < len(s_nc):
            c = s_nc[i]; depth += (c == '{') - (c == '}'); i += 1
        body = s_nc[m.end():i]
        w = re.search(r'\*info\s*=\s*0\b|info\[0\]\s*=\s*0\b|\*info\s*=\s*MAGMA_SUCCESS', body)
        r = re.search(r'\*info\s*(!=|==|<|>)|if\s*\(\s*\*info\s*\)|return\s+\*info', body)
        if r and (not w or w.start() > r.start()):
            line = s_nc[:m.start()].count('\n') + 2
            hits.append((os.path.relpath(f, root), line, name, body[r.start():r.start()+40].replace('\n',' ')))
for h in hits: print('%s:%d %s   first read: %s' % h)
print(len(hits), 'functions')
