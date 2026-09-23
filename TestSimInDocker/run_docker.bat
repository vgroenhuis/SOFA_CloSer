@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

docker image inspect sofa-cube-web >nul 2>&1
if errorlevel 1 (
	echo Image "sofa-cube-web" not found locally - building it now, this can take a minute...
	docker build -t sofa-cube-web .
	if errorlevel 1 (
		echo Build failed.
		pause
		exit /b 1
	)
)

docker container inspect sofa-cube-web >nul 2>&1
if errorlevel 1 (
	echo Creating and starting container "sofa-cube-web"...
	docker run -d -p 8001:8000 --name sofa-cube-web sofa-cube-web
) else (
	for /f "usebackq delims=" %%R in (`docker container inspect -f "{{.State.Running}}" sofa-cube-web`) do set "IS_RUNNING=%%R"
	if /i "!IS_RUNNING!"=="true" (
		echo Container "sofa-cube-web" is already running.
	) else (
		echo Starting existing container "sofa-cube-web"...
		docker start sofa-cube-web >nul
	)
)

echo.
echo Open http://127.0.0.1:8001/ in your browser.
pause
