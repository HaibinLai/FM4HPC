#!/usr/bin/env python3
"""Round 52: make a fuzzable copy of MAGMA multi-GPU sources (see fuzz_hooks.h).
usage: gen_fuzz.py <outdir> <src.cpp> [<src.cpp> ...] -r name=newname [...]
Every statement `call(...);` whose arguments contain a queue expression (queues[..][..], queues[..], queue) gets
`emu_point(<that expression>);` both before and after it, except queue create / destroy / sync and event calls.
A point before an operation delays that operation alone; a point after it also delays an event recorded right after
it (round 52: with after-points only, the potrf positive control M255 was missed, because the delay landed before
`magma_event_record` and every consumer of the event waited with it).  The statement must
end the line (multi-line calls are handled by tracking parentheses).  Prints the number of points inserted."""
import sys, re, pathlib
args = sys.argv[1:]; out = pathlib.Path(args[0]); srcs = []; ren = {}
i = 1
while i < len(args):
    if args[i] == '-r': a, b = args[i + 1].split('='); ren[a] = b; i += 2
    else: srcs.append(pathlib.Path(args[i])); i += 1
QEXPR = re.compile(r'((?:\b\w+\s*(?:->|\.)\s*)*\b(?:queues\s*\[[^\]]+\]\s*\[[^\]]+\]|queues\s*\[[^\]]+\]|queue))\s*\)\s*;\s*(//.*)?$')   # optional struct prefix: data->queues[d], data.queues[d]
SKIP = re.compile(r'magma_queue_(create|destroy|sync)|magma_event_|trace_|magma_xerbla|return\b')
for src in srcs:
    lines = src.read_text().split('\n'); outl = []; n = 0; buf = []; depth = 0; inc_done = False; start = 0; in_c = False
    for ln in lines:
        outl.append(ln)
        if not inc_done and ln.startswith('#include "magma_internal.h"'):
            outl.append('#include "fuzz_hooks.h"'); inc_done = True; continue
        # code part of the line: drop /* */ comments (tracked across lines), string literals and // comments,
        # so that parentheses in comments cannot desynchronise the statement tracking
        code = ''; k = 0
        while k < len(ln):
            if in_c:
                e = ln.find('*/', k)
                if e < 0: k = len(ln)
                else: in_c = False; k = e + 2
            elif ln.startswith('/*', k): in_c = True; k += 2
            elif ln.startswith('//', k): break
            elif ln[k] == '"':
                e = k + 1
                while e < len(ln) and ln[e] != '"': e += 2 if ln[e] == '\\' else 1
                code += '""'; k = e + 1
            else: code += ln[k]; k += 1
        s = '' if code.lstrip().startswith('#') else code   # preprocessor lines (e.g. unbalanced OpenCL macros) are not statements
        if depth == 0 and not re.search(r'\w\s*\(', s): continue
        if not buf: start = len(outl) - 1
        buf.append(s); depth += s.count('(') - s.count(')')
        if depth <= 0:
            stmt = ' '.join(buf); buf = []; depth = 0
            m = QEXPR.search(stmt.strip())
            if m and not SKIP.search(stmt) and stmt.strip().endswith(';') and not stmt.strip().startswith(('if', 'for', 'while', '#')):
                ind = re.match(r'\s*', outl[start]).group(0)
                outl.insert(start, f'{ind}emu_point({m.group(1)});')
                outl.append(f'{ind}emu_point({m.group(1)});'); n += 2
    t = '\n'.join(outl)
    assert inc_done, src
    for a, b in ren.items(): t = re.sub(r'\b%s\b' % a, b, t)
    decl = ''.join(f'extern "C" decltype({a}) {b};\n' for a, b in ren.items())   # prototypes from the MAGMA headers
    t = t.replace('#include "fuzz_hooks.h"', '#include "fuzz_hooks.h"\n' + decl, 1)
    (out / ('fuzz_' + src.name)).write_text(t); print(src.name, 'points', n)
