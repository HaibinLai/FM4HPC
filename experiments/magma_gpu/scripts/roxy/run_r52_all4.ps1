# Round 52, fourth batch: zgehrd_m (with zlahr2_m, zlahru_m); full even targets at n = 96, 300 random before-points at n = 256
$s = 'D:\lean4hpc\magma\run_r52.ps1'
& $s -r gehrd -n 96  -gl '1,2,3,4' -tag e_gehrd96 -step 2
& $s -r gehrd -n 256 -gl '1,2,3,4' -tag e_gehrd256 -sample 300
Set-Content D:\lean4hpc\magma\sweep52\ALL4_DONE.txt 'done'
