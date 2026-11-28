#!/usr/bin/env python3
"""Make emu_[zcds]potrf3_mgpu.cpp from MAGMA 2.10.0 src/[zcds]potrf3_mgpu.cpp: rename the entry point and include
emu_hooks.h after the MAGMA headers.  No other change.  usage: gen_emu.py <magma-2.10.0/src> <outdir>"""
import sys, pathlib
src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
for p in 'zcds':
    t = (src / f'{p}potrf3_mgpu.cpp').read_text()
    assert t.count('#include "trace.h"\n') == 1 and t.count(f'magma_{p}potrf3_mgpu(') == 1
    t = t.replace(f'#define PRECISION_{p}\n', f'#define PRECISION_{p}\n#include "emu_hooks.h"\n', 1)
    t = t.replace(f'magma_{p}potrf3_mgpu(', f'emu_{p}potrf3_mgpu(')
    assert '#include "emu_hooks.h"' in t
    (out / f'emu_{p}potrf3_mgpu.cpp').write_text(t)
    print('wrote', out / f'emu_{p}potrf3_mgpu.cpp')
