@echo off
REM Launches BetterFinger.py with the default ImGui GUI (docked
REM Scene Graph/Viewport/Log panels). Its background is a branded backdrop
REM baked into the GUI itself -- BackgroundSetting has no effect on it.
REM Use run_glfw.bat instead if you want the plain light-blue background.
call "%~dp0sofa_env.bat" || exit /b 1
"%SOFA_EXE%" -l SofaPython3 -g imgui "%~dp0BetterFinger.py"
