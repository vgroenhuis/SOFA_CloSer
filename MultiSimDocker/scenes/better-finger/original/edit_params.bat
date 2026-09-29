@echo off
REM Opens the standalone parameter editor (cube size, wall thickness,
REM material, target pressure, ramp time, instant mode). Edits
REM params.json; after Save, press Reload in the SOFA GUI to apply
REM geometry/material changes.
python "%~dp0params_editor.py"
