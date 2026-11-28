# run one probe (bin\<name>.exe) with the CUDA and OpenBLAS DLLs on PATH; print stdout, stderr and the exit code
$cu = 'D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13'
$env:PATH = "$cu\bin\x86_64;$cu\bin;D:\lean4hpc\openblas\bin;" + $env:PATH
$name = $args[0]; $rest = @($args | Select-Object -Skip 1)
$o = "D:\lean4hpc\magma\sweep52\one_$name.out"; $e = "D:\lean4hpc\magma\sweep52\one_$name.err"
$p = Start-Process -FilePath "D:\lean4hpc\magma\bin\$name.exe" -ArgumentList $rest -NoNewWindow -Wait -PassThru -RedirectStandardOutput $o -RedirectStandardError $e
Get-Content $o; Get-Content $e; "EXIT $($p.ExitCode) (0x{0:X8})" -f $p.ExitCode
