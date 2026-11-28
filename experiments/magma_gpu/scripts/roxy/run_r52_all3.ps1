# Round 52, third batch: zunmqr_m, zgeqrf_m, zungqr_m (full, before + after points), zhetrd_mgpu n = 3072 (random before-points)
$s = 'D:\lean4hpc\magma\run_r52.ps1'
& $s -r unmqr  -n 512 -gl '1,2,3,4' -tag c_unmqr512
& $s -r geqrfm -n 256 -gl '1,2,3,4' -tag c_geqrfm256
& $s -r ungqr  -n 256 -gl '1,2,3,4' -tag c_ungqr256
& $s -r hetrd  -n 3072 -gl '2' -tag c_hetrd3072_g2 -sample 300
& $s -r hetrd  -n 3072 -gl '3,4' -tag c_hetrd3072_g34 -sample 150
Set-Content D:\lean4hpc\magma\sweep52\ALL3_DONE.txt 'done'
