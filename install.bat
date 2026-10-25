@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if errorlevel 1 goto error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto error
if not exist ".env" copy /y ".env.example" ".env" >nul
echo.
echo Kutubxonalar tayyor. .env fayliga OPENROUTER_API_KEY kalitingizni yozing.
echo Keyin start.bat setup buyrug'ini, undan keyin start.bat ni bajaring.
pause
exit /b 0
:error
echo O'rnatishda xato yuz berdi. Yuqoridagi xabarni tekshiring.
pause
exit /b 1
