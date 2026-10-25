@echo off
rem Double-click to set up IdeaPlotter: Python environment, packages, the game map and how ideas get written.
rem Safe to run again; finished steps are skipped.
setlocal
cd /d "%~dp0"
title IdeaPlotter setup

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
  echo IdeaPlotter needs Python 3.10 or newer.
  echo Install it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^), then run install.bat again.
  pause
  exit /b 1
)
%PY% install.py %*
pause
