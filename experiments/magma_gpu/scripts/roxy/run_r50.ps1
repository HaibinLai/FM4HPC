# Round 50: potrf3_mgpu with virtual devices (one process per case; direct exe calls).  n = 1024, nb = 64, delay 100 ms.
#  A  baseline, no delay, MAGMA's own h = 2
#  B  dlP diagonal-buffer hazard: delay the c-th stream-2 trsm that reads dlP (kind 5), h = ngpu (no host hazard), fix 0 / 1
#  C  host-workspace hazard: delay the c-th stream-1 copy of the diagonal block into the owner's dlA (kind 6), various h
$cu = 'D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13'
$env:PATH = "$cu\bin\x86_64;$cu\bin;D:\lean4hpc\openblas\bin;" + $env:PATH
$exe = 'D:\lean4hpc\magma\bin\round50_potrf_emu.exe'
$d = 'D:\lean4hpc\magma\sweep50'; New-Item -ItemType Directory -Force $d | Out-Null
$csv = "$d\r50.csv"; $log = "$d\r50.err"; Set-Content $csv 'set,c,exit,prec,uplo,ngpu,n,nb,h,kind,ms,fix,info,relerr,delays,fixwaits,verdict'; Set-Content $log ''
function Run($set, $c, [string[]]$a) {
  $out = & $exe @a 2>> $log; $ec = $LASTEXITCODE
  $line = ($out | Where-Object { $_ -like 'r50,*' }) -join ''
  if ($line) { Add-Content $csv ("$set,$c,$ec," + $line.Substring(4)) } else { Add-Content $csv "$set,$c,$ec,MISSING,$($a -join ' ')" }
}
foreach ($p in 'z','d') { foreach ($u in 'U','L') { foreach ($g in 1,2,3,4) { Run 'A' 0 @($p,$u,"$g",'1024','64','2','0','0','0','1','0') } } }
foreach ($u in 'U','L') { foreach ($g in 2,3,4) { foreach ($f in 0,1) { foreach ($c in 0..9) { Run 'B' $c @('z',$u,"$g",'1024','64',"$g",'5','100',"$f",'1',"$c") } } } }
foreach ($g in 2,3) { foreach ($c in 0..9) { Run 'B' $c @('d','U',"$g",'1024','64',"$g",'5','100','0','1',"$c") } }
foreach ($p in 'z','d') { foreach ($u in 'U','L') { foreach ($gh in '2,1','2,2','3,2','3,3','4,2','4,3','4,4') { $g,$h = $gh.Split(',')
  foreach ($c in 0..5) { Run 'C' $c @($p,$u,$g,'1024','64',$h,'6','100','0','1',"$c") } } } }
Add-Content $csv 'SWEEP50_DONE'
