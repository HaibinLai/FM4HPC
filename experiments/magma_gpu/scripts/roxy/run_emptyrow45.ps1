# Round 45: isolate the illegal-address error on the empty-row system: one process per (solver, preconditioner, format)
$cu = 'D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13'
$env:PATH = "$cu\bin\x86_64;$cu\bin;D:\lean4hpc\openblas\bin;" + $env:PATH
$d = 'D:\lean4hpc\magma\sweep45empty'; New-Item -ItemType Directory -Force $d | Out-Null; Remove-Item "$d\*" -ErrorAction SilentlyContinue
$env:R45_PROB = 'emptyrow'
foreach ($s in 'CG','PCG','BICGSTAB','PBICGSTAB','GMRES','PGMRES','IDR','PIDR','PCGS','PTFQMR') { foreach ($pc in 'NONE','JACOBI','ILU') { foreach ($f in 'CSR','ELL','SELLP') {
  $env:R45_PREC = $pc; $env:R45_FMT = $f
  & D:\lean4hpc\magma\bin\round45_sparse_edge.exe $s 1> "$d\o.txt" 2> "$d\e.txt"
  $row = (Get-Content "$d\o.txt" | Where-Object { $_ -like "$s,*" }) -join ' '
  $st = (Get-Content "$d\e.txt" | Where-Object { $_ -like '*DONE*' }) -join ' '
  Add-Content "$d\status.txt" "$s,$pc,$f,exit=$LASTEXITCODE,row=[$row],$st"
} } }
Add-Content "$d\status.txt" 'EMPTY45_DONE'
