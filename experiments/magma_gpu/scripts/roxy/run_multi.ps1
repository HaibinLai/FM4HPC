# like run_list.ps1, but keeps every stdout line that is not a usage banner (probes that print several rows per process)
param([string]$exe, [string]$list, [string]$out)
$cu = 'D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13'
$env:PATH = "$cu\bin\x86_64;$cu\bin;D:\lean4hpc\openblas\bin;" + $env:PATH
$d = 'D:\lean4hpc\magma\sweep52'; $csv = "$d\$out.csv"; $log = "$d\$out.err"; Set-Content $csv ''; Set-Content $log ''
foreach ($c in Get-Content $list) {
  if (-not $c.Trim()) { continue }
  $a = $c.Trim() -split '\s+'
  $p = Start-Process -FilePath "D:\lean4hpc\magma\bin\$exe.exe" -ArgumentList $a -NoNewWindow -Wait -PassThru -RedirectStandardOutput "$d\$out.o" -RedirectStandardError "$d\$out.e"
  $lines = Get-Content "$d\$out.o" | Where-Object { $_ -and $_ -notmatch '^%' }
  Add-Content $log "== $c (exit $($p.ExitCode))"; Get-Content "$d\$out.e" | Add-Content $log
  if ($lines) { $lines | ForEach-Object { Add-Content $csv "$($p.ExitCode),$_" } } else { Add-Content $csv "$($p.ExitCode),MISSING,$c" }
}
Add-Content $csv 'LIST_DONE'
