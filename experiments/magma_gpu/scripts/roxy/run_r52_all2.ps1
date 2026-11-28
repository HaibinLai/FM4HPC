# Round 52, second pass with before-points (regenerated copies): even targets for the routines swept in pass 1,
# every target for the new CPU-interface routines; getrfmooc on the zgetrf_m fix build (M258 crashes ngpu >= 3 otherwise)
$s = 'D:\lean4hpc\magma\run_r52.ps1'
& $s -r getrf  -n 256  -gl '1,2,3,4' -tag b_getrf256   -step 2
& $s -r geqrf  -n 256  -gl '1,2,3,4' -tag b_geqrf256   -step 2
& $s -r potrfU -n 1024 -gl '2,3,4'   -tag b_potrfU1024 -step 2
& $s -r potrfL -n 1024 -gl '2,3,4'   -tag b_potrfL1024 -step 2
& $s -r hegst  -n 1024 -gl '1,2,3,4' -tag b_hegst1024  -step 2
& $s -r trsm   -n 1024 -gl '1,2,3,4' -tag b_trsm1024   -step 2
& $s -r getrfm -n 256  -gl '1,2,3,4' -tag b_getrfm256
& $s -r getrfmooc -n 256 -gl '1,2,3,4' -tag b_getrfmooc256 -suffix _fix
& $s -r potrfm -n 1024 -gl '1,2,3,4' -tag b_potrfm1024
Set-Content D:\lean4hpc\magma\sweep52\ALL2_DONE.txt 'done'
