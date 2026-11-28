# Round 46: every sparse storage format x edge-case matrix, one process each; exit code and last STEP recorded
$cu = 'D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13'
$env:PATH = "$cu\bin\x86_64;$cu\bin;D:\lean4hpc\openblas\bin;" + $env:PATH
$d = 'D:\lean4hpc\magma\sweep46'; New-Item -ItemType Directory -Force $d | Out-Null; Remove-Item "$d\*" -ErrorAction SilentlyContinue
foreach ($f in 'ELLPACKT','ELL','DENSE','BCSR','CSC','COO','ELLRT','SELLP','ELLD','CSRLIST','CSRD','CSRL','CSRU','CSRCOO','CUCSR','CSR5') { foreach ($m in 'n1','poisson','random','emptyrow','denserow','rect','zero') {
  & D:\lean4hpc\magma\bin\round46_formats.exe $f $m 1> "$d\o.txt" 2> "$d\e.txt"
  $code = $LASTEXITCODE
  Get-Content "$d\o.txt" | Where-Object { $_ -like 'r46,*' } | Add-Content "$d\all.csv"
  $last = (Get-Content "$d\e.txt" | Where-Object { $_ -match '^(STEP|END)' } | Select-Object -Last 1)
  Add-Content "$d\status.txt" "$f,$m,exit=$code,last=$last"
} }
Add-Content "$d\status.txt" 'FMT46_DONE'
