@echo off
SETLOCAL EnableDelayedExpansion

echo ============================================================
echo 🏥 HEALTHCARE FAIRNESS: MULTI-VERSION SETUP
echo ============================================================
echo.
echo NOTE: This setup looks for ANY 64-bit Python (3.9 to 3.13).
echo.

:: 1. Broad search for 64-bit Python using the Launcher
for %%v in (3.13 3.12 3.11 3.10 3.9) do (
    py -%%v-64 --version >nul 2>&1
    if !errorlevel! equ 0 (
        set PYTHON_CMD=py -%%v-64
        echo -> Found 64-bit Python %%v via Launcher.
        goto FOUND_64
    )
)

:: 2. Try default python if it's already 64-bit
python -c "import struct; exit(0 if struct.calcsize('P') * 8 == 64 else 1)" >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python
    echo -> Found 64-bit default Python.
    goto FOUND_64
)

:: 3. Not found - Give instruction for DIFFERENT version
echo [STILL MISSING] 64-bit Python is not found.now
echo.
echo Your current Python is 3.13 (32-bit). YOU CANNOT install 3.13 (64-bit) 
echo alongside it. You MUST install a different version (like 3.12).
echo.
echo 1. Download Python 3.12 (64-bit): 
echo    https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe
echo.
echo 2. RUN INSTALLER: 
echo    - DO NOT check "Add Python to PATH".
echo    - Just click "Install Now".
echo.
echo After installing, run this BAT file again.
pause
exit /b

:FOUND_64
:: 4. Create Virtual Environment
if not exist "venv_64" (
    echo -> Creating isolated 64-bit Environment...
    %PYTHON_CMD% -m venv venv_64
)

:: 5. Activate and Install
echo -> Installing Research dependencies...
call .\venv_64\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo.
echo ============================================================
echo ✅ SYSTEM READY (AUTOMATED PUBLICATION RUN)
echo ============================================================
echo The research pipeline will now run automatically.
echo No menu selection is required.
echo ============================================================
pause
python run.py
echo.
echo ============================================================
echo 🏁 Research Run Complete.
echo ============================================================
pause
