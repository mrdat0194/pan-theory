@echo off
REM ═══════════════════════════════════════════════════════════════════════
REM  build.bat  –  Compile weeping_demon_dsp.dll (Windows / MinGW-w64)
REM
REM  Requirements:
REM    MinGW-w64 g++ on PATH  (winget install MinGW.MinGW or msys2)
REM    Run this script from the WP\ directory.
REM
REM  NOTE: This build does NOT use the original WDEffect.cpp / Knob / 
REM  Graphics / RtAudio.  It compiles only:
REM    - weeping_demon_wrapper.cpp  (C API + includes WDEffect_headless.h)
REM  WDEffect_headless.h contains the full DSP logic extracted from the 
REM  original WDEffect.cpp but stripped of all OpenGL/Knob dependencies.
REM ═══════════════════════════════════════════════════════════════════════

SET WP_DIR=%~dp0

REM  Only the wrapper itself needs to be compiled:
SET SOURCES="%WP_DIR%weeping_demon_wrapper.cpp"

REM  Include path: WP dir only (WDEffect_headless.h lives here)
SET INC=-I"%WP_DIR%"

SET FLAGS=-O2 -std=c++14 -shared -fPIC

SET OUT="%WP_DIR%weeping_demon_dsp.dll"

echo Building weeping_demon_dsp.dll ...
g++ %FLAGS% %INC% %SOURCES% -o %OUT%

IF %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS: %OUT% built.
    echo.
    echo You can now run:
    echo   python weeping_demon_app.py --file guitar.wav
    echo   python weeping_demon_app.py --rt
) ELSE (
    echo.
    echo FAILED.  Common causes:
    echo   - g++ not on PATH  ^(install MinGW-w64^)
    echo   - Missing WDEffect_headless.h  ^(must be in the WP\ folder^)
)
