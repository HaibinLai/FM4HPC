# Round 45 isolation: one process per (solver, problem, format); a CUDA illegal-address error poisons the context,
# so each case gets a fresh process.  Output: D:\lean4hpc\magma\sweep45iso\all.csv (rows) and status.txt (cuda status per process)
$cu = 'D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13'
$env:PATH = "$cu\bin\x86_64;$cu\bin;D:\lean4hpc\openblas\bin;" + $env:PATH
$d = 'D:\lean4hpc\magma\sweep45iso'; New-Item -ItemType Directory -Force $d | Out-Null
Remove-Item "$d\*" -ErrorAction SilentlyContinue
$solvers = 'CG','PCG','BICGSTAB','PBICGSTAB','QMR','PQMR','TFQMR','PTFQMR','GMRES','PGMRES','IDR','PIDR','CGS','PCGS','BICG','PBICG','LSQR','JACOBI','BA','ITERREF','BOMBARDMENT'
$probs = 'n1','identity8','diag64','tridiag2','zero_rhs','emptyrow','denserow'
foreach ($s in $solvers) { foreach ($p in $probs) { foreach ($f in 'CSR','ELL','SELLP') {
  $env:R45_PROB = $p; $env:R45_FMT = $f
  # direct synchronous call (Start-Process + redirected WaitForExit added ~2 min per case on this machine)
  & D:\lean4hpc\magma\bin\round45_sparse_edge.exe $s 1> "$d\o.txt" 2> "$d\e.txt"
  Get-Content "$d\o.txt" | Where-Object { $_ -like "$s,*" } | Add-Content "$d\all.csv"
  $st = (Get-Content "$d\e.txt" | Where-Object { $_ -like '*DONE*' }) -join ' '
  Add-Content "$d\status.txt" "$s,$p,$f,exit=$LASTEXITCODE,$st"
} } }
Add-Content "$d\status.txt" 'ISO45_DONE'
