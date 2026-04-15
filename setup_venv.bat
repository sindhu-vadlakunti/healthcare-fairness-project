@echo off
setlocal enabledelayedexpansion

echo ===================================================================
echo 🚀 Healthcare Fairness ML Project Setup (Windows)
echo ===================================================================

:: 1. Create a virtual environment
if not exist "venv_64" (
    echo 📦 Creating isolated virtual environment ^(venv_64^)...
    python -m venv venv_64
) else (
    echo ✅ Virtual environment ^(venv_64^) already exists.
)

:: 2. Activate the environment
echo 🔄 Activating virtual environment...
call venv_64\Scripts\activate

:: 3. Install dependencies
echo ⬇️ Installing required packages...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ===================================================================
echo 🎉 Setup complete!
echo ===================================================================

:: 4. Optional: Run the pipeline
set /p RUN_NOW="Do you want to run the research pipeline right now? (Y/N): "
if /i "%RUN_NOW%"=="Y" (
    echo ▶️ Running pipeline...
    python run.py
    pause
) else (
    echo.
    echo You can run the project later by executing:
    echo   1. venv_64\Scripts\activate
    echo   2. python run.py
    pause
)
