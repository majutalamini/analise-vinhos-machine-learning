# Estrutura do relatório escrito

Tom acadêmico, português do Brasil, 6–10 páginas. Números sempre do notebook executado.
Tabelas e figuras numeradas, com legenda e referência no texto.

1. **Capa** — instituição, disciplina, título, integrantes, professor, data.
2. **Introdução** — problema (prever a qualidade sensorial a partir da química), objetivo das
   duas partes, visão geral do pipeline.
3. **Dados** — origem (Wine Quality, UCI; Cortez et al., 2009), variáveis, tamanho, distribuição
   do alvo.
4. **Limpeza e pré-processamento** — tabela dos problemas encontrados no CSV e da correção de
   cada um (com contagens); duplicadas; imputação; outliers (IQR + winsorização); padronização;
   codificação de `type`; por que tudo está dentro do pipeline (data leakage).
5. **Análise exploratória** — 3–4 figuras principais (distribuição do alvo, correlação,
   scatter/boxplots das top features) com a leitura de cada uma.
6. **Parte 1 — Regressão** — modelos e justificativa; validação cruzada; GridSearch (grades e
   melhores parâmetros); tabela de teste (RMSE, MAE, R²); gráfico previsto × real; coeficientes
   e importâncias; discussão.
7. **Parte 2 — Classificação** — esquemas de faixas testados e distribuições; escolha da
   métrica; comparação de classificadores × esquemas; balanceamento; GridSearch; resultados de
   teste; matriz de confusão; análise de erros; features decisivas.
8. **Conclusão** — respostas diretas: melhor modelo em cada parte, variáveis mais importantes,
   limitações, trabalhos futuros.
9. **Referências** — Cortez, P. et al. (2009). Modeling wine preferences by data mining from
   physicochemical properties. *Decision Support Systems*, 47(4), 547–553; documentação do
   scikit-learn e do imbalanced-learn.
