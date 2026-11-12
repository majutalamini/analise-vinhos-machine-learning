@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Gerar previsoes de vinhos
if not exist ".venv\Scripts\python.exe" (
  echo O ambiente Python do projeto nao foi encontrado em .venv.
  echo Consulte o README.md para recriar o ambiente.
  pause
  exit /b 1
)
if not exist "dados\processados\vinhos_limpos.csv" (
  echo A base limpa nao foi encontrada. Execute primeiro 1 - Limpar dados.bat.
  pause
  exit /b 1
)
if not exist "modelos\regressao.joblib" (
  echo O modelo de regressao nao foi encontrado. Execute primeiro 2 - Treinar modelos.bat.
  pause
  exit /b 1
)
if not exist "modelos\classificacao.joblib" (
  echo O modelo de classificacao nao foi encontrado. Execute primeiro 2 - Treinar modelos.bat.
  pause
  exit /b 1
)
echo Gerando previsoes de exemplo com a base limpa do projeto...
".venv\Scripts\python.exe" -X utf8 prever.py "dados\processados\vinhos_limpos.csv" "resultados\previsoes_exemplo.csv"
if errorlevel 1 (
  echo.
  echo Ocorreu um erro. Leia a mensagem acima.
  pause
  exit /b 1
)
echo.
echo Concluido! Abra resultados\previsoes_exemplo.csv no Excel.
pause
