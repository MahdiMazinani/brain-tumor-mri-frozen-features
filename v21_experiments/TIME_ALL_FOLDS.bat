@echo off
setlocal
REM Run from the article2 project root. OPTIONAL: five NEW full grids, no final testing.
if not exist "venv311\Scripts\python.exe" (
  echo Run this batch file from the article2 project root.
  exit /b 1
)
if not exist "ADAPTIVE_RUN.py" exit /b 1
for %%F in (01 02 03 04 05) do (
  venv311\Scripts\python.exe v21_experiments\BENCHMARK_TIMING.py grid --project "." --manifest "cv_figshare_02\manifests\fold_%%F" --out "v21_grid_time_fold%%F"
  if errorlevel 1 exit /b 1
)
echo New wall-clock benchmarks complete. Original runs unchanged.
