# Round 31: mixed-precision IR sweep, one process per group, 1800 s limit each
$cu = 'D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13'
$env:PATH = "$cu\bin\x86_64;$cu\bin;D:\lean4hpc\openblas\bin;" + $env:PATH
$d = 'D:\lean4hpc\magma\sweep31'; New-Item -ItemType Directory -Force $d | Out-Null
foreach ($g in 'po','z','ls','ge') {
  $p = Start-Process -FilePath D:\lean4hpc\magma\bin\round31_mixed.exe -ArgumentList $g -RedirectStandardOutput "$d\$g.csv" -RedirectStandardError "$d\$g.err" -NoNewWindow -PassThru
  if (-not $p.WaitForExit(1800000)) { $p.Kill(); Add-Content "$d\progress.txt" "$g HANG" } else { Add-Content "$d\progress.txt" "$g exit=$($p.ExitCode)" }
}
Add-Content "$d\progress.txt" 'SWEEP31_DONE'
