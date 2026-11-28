# Round 52, gehrd_m pinned-work race: original pinned build on the last 200 points of each (n, ngpu), then the patched build
# (fixed_zgehrd_m.cpp.txt) on every before-point at n = 96.  Uses run_list.ps1 (one process per case).
$env:R52_PIN = '1'
$l = 'D:\lean4hpc\magma\run_list.ps1'
& $l -exe round52_mgpu_fuzz_pin    -list D:\lean4hpc\magma\r52_gehrd_tail.txt  -out q_gehrdpin_tail
& $l -exe round52_mgpu_fuzz_pinfix -list D:\lean4hpc\magma\r52_gehrd_fix96.txt -out q_gehrdpinfix96
Set-Content D:\lean4hpc\magma\sweep52\GEHRDPIN_DONE.txt 'done'
