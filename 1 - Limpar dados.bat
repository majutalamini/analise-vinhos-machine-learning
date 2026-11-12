@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Limpar dados de vinhos
if not exist ".venv\Scripts\python.exe" (
  echo O ambiente Python do projeto nao foi encontrado em .venv.
  echo Consulte o README.md para recriar o ambiente.
  pause
  exit /b 1
)
echo Este comando atualiza o CSV limpo e a auditoria de limpeza.
".venv\Scripts\python.exe" -X utf8 limpar_base.py
if errorlevel 1 (
  echo.
  echo Ocorreu um erro. Leia a mensagem acima.
  pause
  exit /b 1
)
echo.
echo Concluido! Veja dados\processados e resultados\auditoria_limpeza.json.
echo Se os dados mudaram, execute 2 - Treinar modelos.bat para atualizar os modelos.
pause
