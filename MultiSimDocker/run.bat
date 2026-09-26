@echo off
rem Builds the scene images (cached, so quick when nothing changed) and
rem (re)starts the orchestrator.
setlocal
cd /d "%~dp0"
if not exist .env (
	echo No .env found - copying .env.example. Edit it to set the admin password!
	copy .env.example .env >nul
)
call build_scenes.bat || (echo Scene build failed. & pause & exit /b 1)
docker compose up -d --build || (echo Starting the orchestrator failed. & pause & exit /b 1)
for /f "tokens=2 delims==" %%P in ('findstr /b "MSD_PORT=" .env') do set MSD_PORT=%%P
if "%MSD_PORT%"=="" set MSD_PORT=8080
echo.
echo Lobby: http://127.0.0.1:%MSD_PORT%/
echo Admin: http://127.0.0.1:%MSD_PORT%/admin
pause
