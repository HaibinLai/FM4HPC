# Round 29: one process per sparse solver, 600 s limit each; CSV per solver in D:\lean4hpc\magma\sweep29
$cu = 'D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13'
$env:PATH = "$cu\bin\x86_64;$cu\bin;D:\lean4hpc\openblas\bin;" + $env:PATH
$d = 'D:\lean4hpc\magma\sweep29'; New-Item -ItemType Directory -Force $d | Out-Null
$solvers = 'CG','PCG','BICGSTAB','PBICGSTAB','QMR','PQMR','TFQMR','PTFQMR','GMRES','PGMRES','IDR','PIDR','CGS','PCGS','BICG','PBICG','LSQR','JACOBI','BA','ITERREF','BOMBARDMENT'
foreach ($s in $solvers) {
  $p = Start-Process -FilePath D:\lean4hpc\magma\bin\round29_sparse.exe -ArgumentList $s -RedirectStandardOutput "$d\$s.csv" -RedirectStandardError "$d\$s.err" -NoNewWindow -PassThru
  if (-not $p.WaitForExit(600000)) { $p.Kill(); Add-Content "$d\progress.txt" "$s HANG (killed after 600 s)" } else { Add-Content "$d\progress.txt" "$s exit=$($p.ExitCode)" }
}
Add-Content "$d\progress.txt" 'SWEEP29_DONE'
