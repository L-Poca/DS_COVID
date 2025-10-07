@echo off
REM Script de lancement Streamlit DS_COVID - Version Batch Simple
REM ==============================================================

echo Lancement Streamlit DS_COVID...

REM Se placer dans le repertoire du script
cd /d "%~dp0"
echo Repertoire de travail: %CD%

REM Verifier que app.py existe
if not exist "app.py" (
    echo ERREUR: app.py non trouve dans %CD%
    echo Fichiers disponibles:
    dir *.py /b 2>nul
    pause
    exit /b 1
)

echo OK: app.py trouve

REM Trouver l'environnement virtuel
set VENV_PATH=
if exist "..\..\\.venv" set VENV_PATH=..\..\\.venv
if exist "..\\.venv" set VENV_PATH=..\\.venv
if exist ".venv" set VENV_PATH=.venv

if "%VENV_PATH%"=="" (
    echo ERREUR: Environnement virtuel non trouve
    echo Creez avec: python -m venv .venv
    pause
    exit /b 1
)

echo OK: Environnement virtuel trouve dans %VENV_PATH%

REM Activer l'environnement et lancer Streamlit
set ACTIVATE=%VENV_PATH%\Scripts\activate.bat
if exist "%ACTIVATE%" (
    echo Activation et lancement de Streamlit...
    echo URL: http://localhost:8501
    echo Utilisez Ctrl+C pour arreter
    call "%ACTIVATE%" && streamlit run app.py
) else (
    echo ERREUR: Script d'activation non trouve
    pause
    exit /b 1
)

echo Termine.
pause