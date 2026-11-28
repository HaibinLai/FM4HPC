# Round 52: magma_zgetrf_m (fuzzable copy, no delay) with MAGMA_NGR_NB unset / 64 / 128 and ngpu 2..5, one process each
param([string]$suffix = "")
$cu = 'D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13'
$env:PATH = "$cu\bin\x86_64;$cu\bin;D:\lean4hpc\openblas\bin;" + $env:PATH
foreach ($nb in '', '64', '128') { foreach ($g in 2, 3, 4, 5) {
  if ($nb) { $env:MAGMA_NGR_NB = $nb } else { Remove-Item Env:\MAGMA_NGR_NB -ErrorAction SilentlyContinue }
  $p = Start-Process -FilePath D:\lean4hpc\magma\bin\round52_mgpu_fuzz$suffix.exe -ArgumentList 'getrfm', "$g", '256', '-1' -NoNewWindow -Wait -PassThru -RedirectStandardOutput D:\lean4hpc\magma\sweep52\e.out -RedirectStandardError D:\lean4hpc\magma\sweep52\e.err
  $o = (Get-Content D:\lean4hpc\magma\sweep52\e.out | Where-Object { $_ -like 'r52,*' }) -join ''; $w = (Get-Content D:\lean4hpc\magma\sweep52\e.out | Where-Object { $_ -like 'Warning*' }) -join ' '
  "NGR_NB=$(if ($nb) { $nb } else { 'unset' }),ngpu=$g,exit=0x{0:X8},$o,$w" -f $p.ExitCode } }
