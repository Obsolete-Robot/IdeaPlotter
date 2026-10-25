@echo off
rem Double-click to open the IdeaPlotter map. Extra arguments are passed to `ideaplotter serve`,
rem e.g.  run.bat --port 9000   or   run.bat --backend openrouter --model deepseek/deepseek-v4-flash
setlocal
cd /d "%~dp0"
title IdeaPlotter

if not exist ".venv\Scripts\python.exe" (
  echo IdeaPlotter is not set up yet. Double-click install.bat first.
  pause
  exit /b 1
)
if not exist "data\space\xyz.npy" (
  echo The game map is not on this computer yet. Run install.bat again to download it.
  pause
  exit /b 1
)

set PYTHONWARNINGS=ignore
set TRANSFORMERS_VERBOSITY=error
echo Starting IdeaPlotter. Loading the games takes about 30 seconds; your browser opens when it is ready.
echo Keep this window open while you use the map. Close it, or press Ctrl+C, to stop.
echo.
".venv\Scripts\python.exe" -m ideaplotter serve --open %*
if errorlevel 1 pause
