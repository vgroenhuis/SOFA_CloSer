@echo off
REM Launches BetterFinger.py with the older GLFW GUI: a single
REM plain viewport (no docked Scene Graph/Log panels), but it DOES respect
REM the scene's BackgroundSetting -- so the background renders as the
REM plain light-blue color instead of ImGui's branded backdrop.
call "%~dp0sofa_env.bat" || exit /b 1
"%SOFA_EXE%" -l SofaPython3 -g glfw "%~dp0BetterFinger.py"
