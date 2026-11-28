# Round 52: schedule fuzzing sweep.  usage: run_r52.ps1 -r <routine> -n <n> -gl '<ngpu list>' -tag <tag> [-step 1|2] [-suffix _fix]
# -step 2: even targets only (the before-points; each operation's before / after points run back to back)
# -sample K: K random even targets (fixed seed 52) instead of a full sweep, for routines with too many points
# For each ngpu: a dry run gives the number of points P, then one process per target 0..P-1 (delay 100 ms).
param([string]$r, [int]$n, [string]$gl, [string]$tag, [int]$step = 1, [string]$suffix = '', [int]$sample = 0)
$cu = 'D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13'
$env:PATH = "$cu\bin\x86_64;$cu\bin;D:\lean4hpc\openblas\bin;" + $env:PATH
$exe = "D:\lean4hpc\magma\bin\round52_mgpu_fuzz$suffix.exe"
$d = 'D:\lean4hpc\magma\sweep52'; New-Item -ItemType Directory -Force $d | Out-Null
$csv = "$d\$tag.csv"; $log = "$d\$tag.err"; Set-Content $csv 'exit,r52,routine,ngpu,n,target,points,delayed_at,info,rel,verdict'; Set-Content $log ''
foreach ($g in $gl.Split(',')) {
  $dry = (& $exe $r $g $n -1 2>$null | Where-Object { $_ -like 'r52,*' }) -join ''
  Add-Content $csv "$LASTEXITCODE,$dry"
  $P = [int]($dry.Split(',')[5])
  if ($sample -gt 0) { $rng = New-Object System.Random 52; $targets = 1..$sample | ForEach-Object { 2 * $rng.Next(0, [int]($P / 2)) } | Sort-Object -Unique }
  else { $targets = for ($t = 0; $t -lt $P; $t += $step) { $t } }
  foreach ($t in $targets) {
    Add-Content $log "== $r $g $n $t"
    $out = & $exe $r $g $n $t 100 2>> $log; $ec = $LASTEXITCODE
    $line = ($out | Where-Object { $_ -like 'r52,*' }) -join ''
    if ($line) { Add-Content $csv "$ec,$line" } else { Add-Content $csv "$ec,MISSING,$r,$g,$n,$t" }
  }
}
Add-Content $csv 'SWEEP52_DONE'
