# Round 52, pinned-host pass (R52_PIN=1, bin\round52_mgpu_fuzz_pin.exe = _fix sources): CPU-interface routines, before-points (even targets)
$env:R52_PIN = '1'
$s = 'D:\lean4hpc\magma\run_r52.ps1'
& $s -r trsm      -n 1024 -gl '1,2,3,4' -tag p_trsm1024      -step 2 -suffix _pin
& $s -r hegst     -n 1024 -gl '1,2,3,4' -tag p_hegst1024     -step 2 -suffix _pin
& $s -r potrfm    -n 1024 -gl '1,2,3,4' -tag p_potrfm1024    -step 2 -suffix _pin
& $s -r getrfm    -n 256  -gl '1,2,3,4' -tag p_getrfm256     -step 2 -suffix _pin
& $s -r getrfmooc -n 256  -gl '1,2,3,4' -tag p_getrfmooc256  -step 2 -suffix _pin
& $s -r unmqr     -n 512  -gl '1,2,3,4' -tag p_unmqr512      -step 2 -suffix _pin
& $s -r geqrfm    -n 256  -gl '1,2,3,4' -tag p_geqrfm256     -step 2 -suffix _pin
& $s -r ungqr     -n 256  -gl '1,2,3,4' -tag p_ungqr256      -step 2 -suffix _pin
& $s -r gehrd     -n 96   -gl '2,4'     -tag p_gehrd96       -sample 200 -suffix _pin
& $s -r heevd2s   -n 1024 -gl '2,3,4'   -tag p_heevd2s1024   -sample 200 -suffix _pin
& $s -r hetrd     -n 3072 -gl '2'       -tag p_hetrd3072     -sample 100 -suffix _pin
Set-Content D:\lean4hpc\magma\sweep52\PIN_DONE.txt 'done'
