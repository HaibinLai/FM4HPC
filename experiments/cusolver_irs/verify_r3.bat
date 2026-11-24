@echo off
cd /d D:\lean4hpc\irs
for %%S in (11 12 13 14 15) do (
  ..\venv\Scripts\python.exe irs_verify.py R3 %%S cusolverDnDBgesv 40
  ..\venv\Scripts\python.exe irs_verify.py R3 %%S cusolverDnDXgesv 40
  ..\venv\Scripts\python.exe irs_verify.py R3 %%S cusolverDnDSgesv 40
  ..\venv\Scripts\python.exe irs_verify.py R3 %%S cusolverDnDSgesv -60
  ..\venv\Scripts\python.exe irs_verify.py R3 %%S cusolverDnDHgesv 4
)
