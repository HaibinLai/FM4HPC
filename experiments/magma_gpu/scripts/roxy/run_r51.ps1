# Round 51: one process per case, library build (round51_stream.exe) and patched build (round51_stream_fix.exe).  Output sweep51\r51.csv
$cu = 'D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13'
$env:PATH = "$cu\bin\x86_64;$cu\bin;D:\lean4hpc\openblas\bin;" + $env:PATH
$d = 'D:\lean4hpc\magma\sweep51'; New-Item -ItemType Directory -Force $d | Out-Null
$csv = "$d\r51.csv"; $log = "$d\r51.err"; Set-Content $csv 'build,exit,r51,mode,a1,a2,a3,observed,expected,verdict'; Set-Content $log ''
function Run($b, [string[]]$a) {
  $exe = "D:\lean4hpc\magma\bin\round51_stream$b.exe"; Add-Content $log "== $b $($a -join ' ')"
  $out = & $exe @a 2>> $log; $ec = $LASTEXITCODE
  $line = ($out | Where-Object { $_ -like 'r51,*' }) -join ''
  $tag = if ($b) { 'fix' } else { 'lib' }
  if ($line) { Add-Content $csv "$tag,$ec,$line" } else { Add-Content $csv "$tag,$ec,MISSING,$($a -join ' ')" }
}
foreach ($b in '', '_fix') {
  foreach ($p in 'd','z') { foreach ($sh in 'bidiag','random') { foreach ($m in 1000, 20000) { foreach ($rep in 1..3) { Run $b @('syncfree',$p,$sh,"$m") } } } }
  foreach ($k in 0..3) { Run $b @('h2batched',"$k") }
  foreach ($q in 'magma','blocking','nonblocking') { Run $b @('flag','slag2h',$q,'0'); Run $b @('flag','slag2h_ok',$q,'0') }
}
foreach ($r in 'zlag2c','dlag2s','zlat2c','dlat2s') { foreach ($q in 'magma','blocking','nonblocking') { foreach ($dl in 0, 100) { Run '' @('flag',$r,$q,"$dl") } } }
foreach ($q in 'magma','blocking','nonblocking') { foreach ($busy in 0, 150) { Run '' @('trtri',$q,"$busy") } }
Add-Content $csv 'SWEEP51_DONE'
