# run_list.ps1 with one environment variable set first: run_list_env.ps1 -var NAME -val VALUE -exe ... -list ... -out ...
param([string]$var, [string]$val, [string]$exe, [string]$list, [string]$out)
Set-Item -Path "env:$var" -Value $val
& D:\lean4hpc\magma\run_list.ps1 -exe $exe -list $list -out $out
