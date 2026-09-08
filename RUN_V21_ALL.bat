@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"

REM Put this BAT in the article2 root, beside v21_experiments and venv311.
REM All stages run sequentially. Original results are never overwritten.
REM Full grid timing is EXPENSIVE: five fresh development grids, no test scoring.
REM Set RUN_FULL_GRID_TIMING=0 before starting to omit only that timing benchmark.
set "RUN_FULL_GRID_TIMING=1"
set "PY=%CD%\venv311\Scripts\python.exe"
set "SCRIPTS=%CD%\v21_experiments"
set "PLAN=%CD%\cv_figshare_02"
set "MAN7023=%CD%\prepared\dataset_02"
set "RUN7023=%CD%\runs_adaptive\experiment_02"
set "DEVICE=cuda"
set "BATCH=16"
set "EPOCHS=20"
set "PATIENCE=5"
set "REPEATS=3"
set "OUT="
set "STEP=preflight"

if not exist "%PY%" goto :missing
for %%S in (NEW_CONTROLS.py EXPERIMENT_COMMON.py VISION_HELPERS.py OOF_ANALYSIS.py MERGE_OOF.py BENCHMARK_TIMING.py test_v21.py) do (
  if not exist "%SCRIPTS%\%%S" goto :missing
)
if not exist "%PLAN%\plan.json" goto :missing
if not exist "%PLAN%\cv_oof_predictions.csv" goto :missing
if not exist "%MAN7023%\dataset.json" goto :missing
if not exist "%RUN7023%\frozen_selection.json" goto :missing
if "%RUN_FULL_GRID_TIMING%"=="1" if not exist "%CD%\ADAPTIVE_RUN.py" goto :missing
for %%F in (01 02 03 04 05) do (
  if not exist "%PLAN%\manifests\fold_%%F\dataset.json" goto :missing
  if not exist "%PLAN%\runs\fold_%%F\frozen_selection.json" goto :missing
)
set "STAMP="
for /f %%T in ('powershell.exe -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%T"
if not defined STAMP set "STAMP=run"
set "OUT=%CD%\v21_results_%STAMP%_%RANDOM%"
if exist "%OUT%" goto :collision
mkdir "%OUT%\logs"
if errorlevel 1 goto :failed
copy /y "%~f0" "%OUT%\RUN_V21_ALL_used.bat" >nul
> "%OUT%\RUN_INFO.txt" echo Started: %DATE% %TIME%
>> "%OUT%\RUN_INFO.txt" echo Project: %CD%
>> "%OUT%\RUN_INFO.txt" echo Output: %OUT%
>> "%OUT%\RUN_INFO.txt" echo Full grid timing: %RUN_FULL_GRID_TIMING%
>> "%OUT%\RUN_INFO.txt" echo New supplementary exploratory experiments. No deduplication or synthetic imbalance.
>> "%OUT%\RUN_INFO.txt" echo No historical independence is established by these new runs.

echo ============================================================
echo ALL NEW V21 EXPERIMENTS - SEQUENTIAL EXECUTION
echo Output: %OUT%
echo Full grid timing enabled: %RUN_FULL_GRID_TIMING%
echo ============================================================
set "STEP=01_environment"
call :run -c "import sys,json,torch,torchvision,sklearn,numpy,pandas,scipy,joblib; print(sys.version); print(json.dumps({'torch':torch.__version__,'torchvision':torchvision.__version__,'sklearn':sklearn.__version__,'numpy':numpy.__version__,'CUDA':torch.cuda.is_available()},indent=2)); assert torch.cuda.is_available(), 'CUDA unavailable; inspect venv311 before training'"
if errorlevel 1 goto :failed
set "STEP=02_tests"
call :run -m unittest discover -s "%SCRIPTS%" -p test_v21.py -v
if errorlevel 1 goto :failed
set "STEP=03_original_oof_statistics"
call :run "%SCRIPTS%\OOF_ANALYSIS.py" --oof "%PLAN%\cv_oof_predictions.csv" --out "%OUT%\original_oof_analysis" --resamples 2000
if errorlevel 1 goto :failed
set "STEP=04_historical_timing_cv"
call :run "%SCRIPTS%\BENCHMARK_TIMING.py" audit --root "%PLAN%" --out "%OUT%\historical_timing_cv"
if errorlevel 1 goto :failed
set "STEP=05_historical_timing_7023"
call :run "%SCRIPTS%\BENCHMARK_TIMING.py" audit --root "%RUN7023%" --out "%OUT%\historical_timing_7023"
if errorlevel 1 goto :failed
set "STEP=06_pretrained_weights"
call :run "%SCRIPTS%\BENCHMARK_TIMING.py" weights --download
if errorlevel 1 goto :failed
set "STEP=07_init_lp_cv"
call :run "%SCRIPTS%\NEW_CONTROLS.py" init --kind lp --plan "%PLAN%" --out "%OUT%\lp_cv"
if errorlevel 1 goto :failed
set "STEP=08_init_resnet18_cv"
call :run "%SCRIPTS%\NEW_CONTROLS.py" init --kind e2e --plan "%PLAN%" --out "%OUT%\resnet18_cv" --epochs %EPOCHS% --patience %PATIENCE% --batch %BATCH% --device %DEVICE%
if errorlevel 1 goto :failed
set "STEP=09_init_lp_7023"
call :run "%SCRIPTS%\NEW_CONTROLS.py" init --kind lp --manifest "%MAN7023%" --run "%RUN7023%" --out "%OUT%\lp_7023"
if errorlevel 1 goto :failed
set "STEP=10_develop_lp_cv"
call :run "%SCRIPTS%\NEW_CONTROLS.py" develop --out "%OUT%\lp_cv"
if errorlevel 1 goto :failed
set "STEP=11_develop_resnet18_cv"
call :run "%SCRIPTS%\NEW_CONTROLS.py" develop --out "%OUT%\resnet18_cv"
if errorlevel 1 goto :failed
set "STEP=12_develop_lp_7023"
call :run "%SCRIPTS%\NEW_CONTROLS.py" develop --out "%OUT%\lp_7023"
if errorlevel 1 goto :failed
if "%RUN_FULL_GRID_TIMING%"=="1" (
  for %%F in (01 02 03 04 05) do (
    set "STEP=13_grid_timing_fold_%%F"
    call :run "%SCRIPTS%\BENCHMARK_TIMING.py" grid --project "%CD%" --manifest "%PLAN%\manifests\fold_%%F" --out "%OUT%\grid_timing\fold_%%F"
    if errorlevel 1 goto :failed
  )
)
set "STEP=14_evaluate_lp_cv"
call :run "%SCRIPTS%\NEW_CONTROLS.py" evaluate --out "%OUT%\lp_cv"
if errorlevel 1 goto :failed
set "STEP=15_evaluate_resnet18_cv"
call :run "%SCRIPTS%\NEW_CONTROLS.py" evaluate --out "%OUT%\resnet18_cv"
if errorlevel 1 goto :failed
set "STEP=16_evaluate_lp_7023"
call :run "%SCRIPTS%\NEW_CONTROLS.py" evaluate --out "%OUT%\lp_7023"
if errorlevel 1 goto :failed
set "STEP=17_merge_cv_oof"
call :run "%SCRIPTS%\MERGE_OOF.py" --inputs "%PLAN%\cv_oof_predictions.csv" "%OUT%\lp_cv\new_controls_oof.csv" "%OUT%\resnet18_cv\new_controls_oof.csv" --out "%OUT%\combined_cv\oof.csv"
if errorlevel 1 goto :failed
set "STEP=18_combined_cv_statistics"
call :run "%SCRIPTS%\OOF_ANALYSIS.py" --oof "%OUT%\combined_cv\oof.csv" --out "%OUT%\combined_cv\statistics" --resamples 2000
if errorlevel 1 goto :failed
set "STEP=19_lp_pair_statistics"
call :run "%SCRIPTS%\OOF_ANALYSIS.py" --oof "%OUT%\lp_cv\new_controls_oof.csv" --out "%OUT%\lp_cv\posthoc_pooling" --a lp_tuned_raw --b lp_tuned_pca95 --resamples 2000
if errorlevel 1 goto :failed
set "STEP=20_extraction_figshare"
call :run "%SCRIPTS%\BENCHMARK_TIMING.py" extraction --manifest "%PLAN%\manifests\fold_01" --out "%OUT%\extraction_figshare" --device %DEVICE% --batch %BATCH% --repeats %REPEATS%
if errorlevel 1 goto :failed
set "STEP=21_extraction_7023"
call :run "%SCRIPTS%\BENCHMARK_TIMING.py" extraction --manifest "%MAN7023%" --out "%OUT%\extraction_7023" --device %DEVICE% --batch %BATCH% --repeats %REPEATS%
if errorlevel 1 goto :failed
> "%OUT%\ALL_COMPLETED.txt" echo All requested stages completed: %DATE% %TIME%
echo SUCCESS. Results and logs: %OUT%
pause
exit /b 0
:run
echo [%DATE% %TIME%] %STEP%
"%PY%" -u %* > "%OUT%\logs\%STEP%.log" 2>&1
if errorlevel 1 exit /b 1
exit /b 0
:failed
echo STOPPED at: %STEP%
if defined OUT > "%OUT%\STOPPED.txt" echo Failed stage: %STEP% at %DATE% %TIME%
pause
exit /b 1
:missing
echo Required file or directory is missing. Put this BAT in the article2 project root.
pause
exit /b 1
:collision
echo Output folder already exists. No existing folder was modified.
pause
exit /b 1
