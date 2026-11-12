@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Treinar modelos de vinhos
if not exist ".venv\Scripts\python.exe" (
  echo O ambiente Python do projeto nao foi encontrado em .venv.
  echo Consulte o README.md para recriar o ambiente.
  pause
  exit /b 1
)
if not exist "dados\processados\vinhos_limpos.csv" (
  echo Base limpa nao encontrada. Execute primeiro 1 - Limpar dados.bat.
  pause
  exit /b 1
)
echo Este comando usa a base limpa e atualiza os modelos e as metricas.
echo O treinamento pode demorar alguns minutos.
echo.
".venv\Scripts\python.exe" -X utf8 treinar.py
if errorlevel 1 (
  echo.
  echo Ocorreu um erro. Leia a mensagem acima.
  pause
  exit /b 1
)
echo.
echo Concluido! Veja as pastas modelos e resultados.
pause
