@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Gerar relatorio de vinhos
if not exist ".venv\Scripts\python.exe" (
  echo O ambiente Python do projeto nao foi encontrado em .venv.
  echo Consulte o README.md para recriar o ambiente.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -X utf8 gerar_relatorio.py
if errorlevel 1 (
  echo.
  echo Ocorreu um erro. Leia a mensagem acima.
  pause
  exit /b 1
)
echo.
echo Abrindo o relatorio no navegador. Para salvar em PDF, use Imprimir no navegador.
start "" "%~dp0resultados\relatorio.html"
pause
