# run bin\<exe>.exe once per line of a case list (one process per case); stdout rows go to <out>.csv with the exit code first,
# a missing row is written as MISSING; stderr goes to <out>.err with an "== case" header.  Ends with LIST_DONE.
param([string]$exe, [string]$list, [string]$out)
$cu = 'D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13'
$env:PATH = "$cu\bin\x86_64;$cu\bin;D:\lean4hpc\openblas\bin;" + $env:PATH
$d = 'D:\lean4hpc\magma\sweep52'; $csv = "$d\$out.csv"; $log = "$d\$out.err"; Set-Content $csv 'exit,row'; Set-Content $log ''
foreach ($c in Get-Content $list) {
  if (-not $c.Trim()) { continue }
  $a = $c.Trim() -split '\s+'
  $p = Start-Process -FilePath "D:\lean4hpc\magma\bin\$exe.exe" -ArgumentList $a -NoNewWindow -Wait -PassThru -RedirectStandardOutput "$d\$out.o" -RedirectStandardError "$d\$out.e"
  $line = Get-Content "$d\$out.o" | Where-Object { $_ -match '^r[0-9]' } | Select-Object -First 1
  Add-Content $log "== $c"; Get-Content "$d\$out.e" | Add-Content $log
  if ($line) { Add-Content $csv "$($p.ExitCode),$line" } else { Add-Content $csv "$($p.ExitCode),MISSING,$c" }
}
Add-Content $csv 'LIST_DONE'
