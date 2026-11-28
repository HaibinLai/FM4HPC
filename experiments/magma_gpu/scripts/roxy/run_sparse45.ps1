# Round 45: sparse edge-case systems, one process per solver, 120 s limit each; CSV per solver in D:\lean4hpc\magma\sweep45
$cu = 'D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13'
$env:PATH = "$cu\bin\x86_64;$cu\bin;D:\lean4hpc\openblas\bin;" + $env:PATH
$d = 'D:\lean4hpc\magma\sweep45'; New-Item -ItemType Directory -Force $d | Out-Null
$solvers = 'CG','PCG','BICGSTAB','PBICGSTAB','QMR','PQMR','TFQMR','PTFQMR','GMRES','PGMRES','IDR','PIDR','CGS','PCGS','BICG','PBICG','LSQR','JACOBI','BA','ITERREF','BOMBARDMENT'
foreach ($s in $solvers) {
  $p = Start-Process -FilePath D:\lean4hpc\magma\bin\round45_sparse_edge.exe -ArgumentList $s -RedirectStandardOutput "$d\$s.csv" -RedirectStandardError "$d\$s.err" -NoNewWindow -PassThru
  if (-not $p.WaitForExit(120000)) { $p.Kill(); Add-Content "$d\progress.txt" "$s HANG (killed after 120 s)" } else { Add-Content "$d\progress.txt" "$s exit=$($p.ExitCode)" }
}
Add-Content "$d\progress.txt" 'SWEEP45_DONE'
