@echo off
rem Builds the Docker image for every scene in scenes\<id>\ that has a
rem Dockerfile, tagged msd-scene-<id> (what the orchestrator expects unless a
rem scene.json sets "image" explicitly). Pass scene ids to build only those.
setlocal
cd /d "%~dp0"
if "%~1"=="" (
	for /d %%S in (scenes\*) do call :build %%~nxS || exit /b 1
) else (
	for %%S in (%*) do call :build %%S || exit /b 1
)
exit /b 0

:build
if not exist "scenes\%1\Dockerfile" exit /b 0
if not exist "scenes\%1\scene.json" exit /b 0
echo === Building msd-scene-%1
docker build -t msd-scene-%1 scenes\%1
exit /b %errorlevel%
