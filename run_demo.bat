@echo off
rem Demand Predictor - DEMO copy. Double-click to start.
rem Uses its own database (data\demo.db) and port 8502, so your own data in data\planner.db is not touched.
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
if not exist ".venv\Scripts\python.exe" (
  echo First run: setting up. This takes a few minutes and needs internet access.
  %PY% -m venv .venv || goto :nopython
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
)
echo Checking packages...
".venv\Scripts\python.exe" -m pip install -q -r requirements.txt || goto :piperror
set PLANNER_DB=%~dp0data\demo.db
set PLANNER_DEMO=1
echo Starting the DEMO. Your browser will open at http://localhost:8502
echo Keep this window open while you use the app. Close it to stop.
start "" /b cmd /c "timeout /t 5 >nul & start http://localhost:8502"
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8502
goto :eof
:nopython
echo.
echo Python was not found. Install Python 3.10 or newer from python.org (tick "Add python.exe to PATH") and run this again.
pause
goto :eof
:piperror
echo.
echo The packages could not be installed. If you are behind a company proxy, ask IT for the pip proxy setting,
echo then delete the .venv folder and run this again.
pause
