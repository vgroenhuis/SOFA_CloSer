@echo off
rem Builds the Docker image for every scene in scenes\<id>\ that has a
rem Dockerfile, tagged msd-scene-<id> (what the orchestrator expects unless a
rem scene.json sets "image" explicitly). Pass scene ids to build only those.
rem The shared base image for sofaweb scenes (scenes\_base -> msd-sofa-base)
rem is built first; Docker's cache makes that quick when nothing changed.
setlocal
cd /d "%~dp0"
if exist "scenes\_base\Dockerfile" (
	echo === Building msd-sofa-base
	docker build -t msd-sofa-base scenes\_base || exit /b 1
)
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
