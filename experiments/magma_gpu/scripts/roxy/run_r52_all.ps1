# Round 52: all schedule-fuzzing sweeps after getrf, in series (each writes sweep52\<tag>.csv)
$s = 'D:\lean4hpc\magma\run_r52.ps1'
& $s -r potrfU -n 1024 -gl '2,3,4' -tag potrfU1024
& $s -r potrfL -n 1024 -gl '2,3,4' -tag potrfL1024
& $s -r geqrf -n 256 -gl '1,2,3,4' -tag geqrf256
& $s -r hegst -n 1024 -gl '1,2,3,4' -tag hegst1024
& $s -r trsm -n 1024 -gl '1,2,3,4' -tag trsm1024
& $s -r potrfU -n 4096 -gl '2' -tag potrfU4096   # positive control: 16 blocks, M255 reachable
Set-Content D:\lean4hpc\magma\sweep52\ALL_DONE.txt 'done'
