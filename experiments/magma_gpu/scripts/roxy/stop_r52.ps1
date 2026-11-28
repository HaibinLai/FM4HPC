# stop the round-52 sweep (only processes started by run_r52*.ps1 / round52_mgpu_fuzz.exe)
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*run_r52*' -or $_.Name -eq 'round52_mgpu_fuzz.exe' } | ForEach-Object { "stop $($_.ProcessId) $($_.Name)"; Stop-Process -Id $_.ProcessId -Force }
