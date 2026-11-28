# run one round-52 case and print its stdout, stderr and exit code
$cu = 'D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13'
$env:PATH = "$cu\bin\x86_64;$cu\bin;D:\lean4hpc\openblas\bin;" + $env:PATH
$p = Start-Process -FilePath D:\lean4hpc\magma\bin\round52_mgpu_fuzz.exe -ArgumentList $args -NoNewWindow -Wait -PassThru -RedirectStandardOutput D:\lean4hpc\magma\sweep52\one.out -RedirectStandardError D:\lean4hpc\magma\sweep52\one.err
Get-Content D:\lean4hpc\magma\sweep52\one.out; Get-Content D:\lean4hpc\magma\sweep52\one.err; "EXIT $($p.ExitCode) (0x{0:X8})" -f $p.ExitCode
