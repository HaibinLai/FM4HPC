$cu = 'D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13'
$env:PATH = "$cu\bin\x86_64;$cu\bin;D:\lean4hpc\openblas\bin;" + $env:PATH
foreach ($who in 'lapack','magma') { foreach ($rt in 'gesvd','gesdd') { foreach ($job in 'N','S') {
  $out = "D:\lean4hpc\magma\sweep\svdinf_$who`_$rt`_$job.txt"
  $p = Start-Process -FilePath D:\lean4hpc\magma\bin\round27_focus.exe -ArgumentList "svdinf $who $rt $job" -RedirectStandardOutput $out -NoNewWindow -PassThru
  if (-not $p.WaitForExit(30000)) { $p.Kill(); "$who $rt job=$job : HANG (killed after 30 s)" } else { "$who $rt job=$job : " + ((Get-Content $out) -join ' | ') }
}}}
