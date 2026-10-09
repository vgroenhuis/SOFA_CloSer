@echo off
REM Shared helper: resolves SOFA_EXE (the full path to runSofa.exe) for this
REM machine. Called by run_imgui.bat / run_glfw.bat / run_sofa.bat so the
REM install path only needs to be set once per machine, not hardcoded per
REM script.
REM
REM One-time setup on a new machine -- pick ONE of:
REM   1) Persistent (recommended): open a terminal and run
REM        setx SOFA_ROOT "C:\path\to\SOFA\v26.06.00"
REM      then open a NEW terminal window (setx only affects future sessions).
REM   2) Persistent via GUI: System Properties -> Environment Variables ->
REM      add a User variable SOFA_ROOT pointing at your SOFA install folder
REM      (the one containing "bin\runSofa.exe").
REM   3) One-off override for a single run: set SOFA_ROOT first, e.g.
REM        set SOFA_ROOT=D:\SOFA\v26.06.00 & run_imgui.bat
REM Alternatively set SOFA_EXE directly to the full path of runSofa.exe if
REM your install doesn't follow the "<root>\bin\runSofa.exe" layout.

if defined SOFA_ROOT set "SOFA_EXE=%SOFA_ROOT%\bin\runSofa.exe"

REM Fall back to the current user's own home directory (works on any
REM machine/account following the same <home>\SOFA\v26.06.00 layout)
REM only if nothing else was set.
if not defined SOFA_EXE set "SOFA_EXE=%USERPROFILE%\SOFA\v26.06.00\bin\runSofa.exe"

if not exist "%SOFA_EXE%" (
    echo.
    echo [sofa_env.bat] Could not find runSofa.exe at:
    echo   %SOFA_EXE%
    echo.
    echo Set the SOFA_ROOT environment variable to your SOFA install folder
    echo ^(the one containing bin\runSofa.exe^), e.g.:
    echo   setx SOFA_ROOT "C:\path\to\SOFA\v26.06.00"
    echo then open a new terminal and try again. See comments in this file
    echo for details.
    echo.
    exit /b 1
)

REM The SofaPython3 plugin (needed to load .py scenes) ships bindings built
REM for a specific CPython ABI -- v26.06.00 uses cp312 (Python 3.12, 64-bit).
REM If that exact Python isn't installed/on PATH, runSofa loads and starts
REM fine but silently fails to load SofaPython3.dll, so -g imgui/glfw comes
REM up with no scene. This is just a heads-up; it doesn't block the launch.
where python312.dll >nul 2>&1
if errorlevel 1 (
    echo.
    echo [sofa_env.bat] WARNING: python312.dll not found on PATH.
    echo SofaPython3 ^(needed to load .py scenes like PneuNetFinger.py^)
    echo requires Python 3.12, 64-bit, installed and on PATH. Install it from
    echo https://www.python.org/downloads/ and re-run.
    echo.
)
