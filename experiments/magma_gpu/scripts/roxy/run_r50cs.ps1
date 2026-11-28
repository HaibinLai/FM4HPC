# Round 50 addendum: c and s precisions (same sets as run_r50.ps1, smaller).  Output sweep50\r50cs.csv
$cu = 'D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13'
$env:PATH = "$cu\bin\x86_64;$cu\bin;D:\lean4hpc\openblas\bin;" + $env:PATH
$exe = 'D:\lean4hpc\magma\bin\round50_potrf_emu.exe'
$d = 'D:\lean4hpc\magma\sweep50'; $csv = "$d\r50cs.csv"; $log = "$d\r50cs.err"
Set-Content $csv 'set,c,exit,prec,uplo,ngpu,n,nb,h,kind,ms,fix,info,relerr,delays,fixwaits,verdict'; Set-Content $log ''
function Run($set, $c, [string[]]$a) {
  $out = & $exe @a 2>> $log; $ec = $LASTEXITCODE
  $line = ($out | Where-Object { $_ -like 'r50,*' }) -join ''
  if ($line) { Add-Content $csv ("$set,$c,$ec," + $line.Substring(4)) } else { Add-Content $csv "$set,$c,$ec,MISSING,$($a -join ' ')" }
}
foreach ($p in 'c','s') { foreach ($u in 'U','L') { Run 'A' 0 @($p,$u,'2','1024','64','2','0','0','0','1','0') } }
foreach ($u in 'U','L') { foreach ($f in 0,1) { foreach ($c in 0..9) { Run 'B' $c @('c',$u,'2','1024','64','2','5','100',"$f",'1',"$c") } } }
foreach ($c in 0..9) { Run 'B' $c @('s','U','2','1024','64','2','5','100','0','1',"$c") }
foreach ($p in 'c','s') { foreach ($gh in '2,1','2,2','3,2','3,3') { $g,$h = $gh.Split(',')
  foreach ($c in 0..5) { Run 'C' $c @($p,'U',$g,'1024','64',$h,'6','100','0','1',"$c") } } }
Add-Content $csv 'SWEEP50CS_DONE'
