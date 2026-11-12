# Roteiro da apresentação (até 10 minutos, presencial)

~10 slides, pouco texto por slide, um gráfico por slide sempre que possível. Divida as falas
entre os integrantes — cada um deve conseguir explicar a parte do código que apresenta.

| # | Slide | Tempo | Conteúdo |
|---|---|---|---|
| 1 | Título | 0:15 | tema, equipe |
| 2 | Problema e dados | 0:45 | o que é `quality`, 11 variáveis + tipo, nº de vinhos |
| 3 | Limpeza | 1:15 | os problemas do CSV e como foram corrigidos (tabela antes → depois) |
| 4 | EDA | 1:00 | distribuição do alvo + correlação (alcohol, density, volatile acidity) |
| 5 | Pipeline | 0:45 | diagrama: imputação → winsorização → padronização → modelo; sem vazamento |
| 6 | Regressão — resultados | 1:15 | tabela RMSE/MAE/R² vs baseline, melhor modelo |
| 7 | Regressão — interpretação | 1:00 | coeficientes / importância; regressão à média |
| 8 | Classificação — faixas e métrica | 1:00 | esquemas testados, desbalanceamento, por que F1-macro |
| 9 | Classificação — resultados | 1:15 | balanceamento, melhor modelo, matriz de confusão |
| 10 | Conclusões | 0:30 | 3 frases de conclusão + limitações |

Perguntas prováveis da banca (preparar resposta curta):
- Por que não usar acurácia? · Por que SMOTE só dentro do pipeline?
- Por que o R² é "baixo"? · Por que residual sugar tem coeficiente positivo?
- Como os valores corrompidos foram corrigidos e como vocês sabem que a correção está certa?
- O que o GridSearch mudou em relação ao modelo padrão?
