---
name: trabalho-ml-qualidade-vinhos
description: Use para criar ou alterar o Trabalho Final de Machine Learning sobre o Wine Quality Dataset (regressão e classificação da nota quality em Python) — notebook com explicações em Markdown, limpeza do CSV da equipe, relatório escrito e roteiro da apresentação.
---

# Trabalho Final de ML — Qualidade de Vinhos (Python)

Projeto acadêmico: pipeline completo de Machine Learning sobre o Wine Quality Dataset
(vinhos tintos e brancos, 11 variáveis físico-químicas + `type`, alvo `quality` de 0 a 10).
Tudo em **Python**, com textos em **português do Brasil**, explicando o *porquê* de cada
decisão — a equipe precisa entender e conseguir explicar o código na apresentação.

## Entregáveis exigidos pelo enunciado
1. **Jupyter Notebook** com explicações em Markdown, código organizado e comentado.
2. **Relatório escrito** (estrutura em `references/relatorio.md`).
3. **Roteiro da apresentação** presencial de até 10 minutos (`references/apresentacao.md`).

## Requisitos do enunciado (checklist — tudo precisa aparecer)
**Parte 1 — Regressão:** EDA (histogramas, scatter plots, matriz de correlação) · ausentes,
padronização, detecção de outliers · ≥ 3 modelos de regressão · RMSE, MAE, R² ·
GridSearchCV/RandomizedSearchCV · interpretação de coeficientes e variáveis mais importantes.

**Parte 2 — Classificação:** discretizar `quality` em faixas e **testar diferentes faixas** ·
verificar desbalanceamento e justificar ajustes · LabelEncoder/OneHotEncoder ·
under/oversampling se necessário · ≥ 3 classificadores · métrica escolhida e justificada ·
GridSearchCV · análise de erros (quais classes se confundem), melhor modelo, features decisivas.

## Arquivos da skill
- `scripts/limpeza.py` — limpeza do CSV (módulo + CLI) e relatório de correções.
- `templates/notebook_template.py` — o notebook inteiro em formato "percent" (`# %%`).
- `scripts/gerar_notebook.py` — transforma o template em `.ipynb`, preenchendo marcadores.
- `references/analise.md` — como interpretar os resultados (leia antes de escrever as análises).
- `references/relatorio.md` e `references/apresentacao.md` — estruturas dos outros entregáveis.

## Fluxo de trabalho

### 1. Inspecionar o CSV da equipe antes de tudo
Cada equipe recebe um arquivo `group_N_winequality.csv` e **ele vem sujo**. No arquivo do
grupo 1 foram encontrados:
- `alcohol` como texto: `'100.333.333.333.333'` (= 10,03, separador de milhar do Excel) e `'9.39.39.39…'`;
- vírgula decimal perdida: `chlorides` 76 → 0,076 (quase a coluna toda), `volatile acidity` 675 → 0,675,
  `density` 9.978 / 99.695 / 997 → 0,9978 / 0,99695 / 0,997;
- outliers isolados impossíveis (pH 87,48; citric acid 19,35; total SO2 2430…);
- `type` = `'whitewhite…'` e um `type` vazio; `quality` = 238 e um `quality` vazio;
- `fixed acidity` com ~150 vazios; ~1.100 linhas duplicadas.

Rode `python scripts/limpeza.py <csv>` e confira o relatório. Depois compare as médias por
`type` com as do dataset original da UCI (tinto: chlorides ≈ 0,087, volatile acidity ≈ 0,53,
density ≈ 0,9967; branco: chlorides ≈ 0,046, volatile acidity ≈ 0,28, density ≈ 0,9940).
**Se o CSV de outra equipe tiver um problema diferente**, ajuste `FAIXAS`/`DIVISOR_INTEIROS`
ou acrescente uma regra **em `scripts/limpeza.py` e na célula de limpeza do template** (as duas
cópias devem ficar iguais) e descreva o problema novo no texto do diagnóstico.

Regras que não mudam: o alvo inválido é **removido**, nunca imputado; imputação e tratamento
de outliers ficam **dentro do pipeline** (ajustados só no treino); SMOTE só dentro do
`imblearn.pipeline.Pipeline`. Explique isso como prevenção de *data leakage*.

### 2. Gerar e executar o notebook
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn nbformat nbconvert ipykernel
python scripts/gerar_notebook.py --csv group_1_winequality.csv --saida trabalho_ml_vinhos.ipynb --equipe "Nomes"
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 trabalho_ml_vinhos.ipynb
```
O CSV precisa estar na mesma pasta do notebook. A execução leva ~4 min em 2 núcleos
(os GridSearch são o grosso). Se der erro, corrija **no template** e gere de novo.

### 3. Escrever as análises com os números reais
O notebook tem marcadores `> ✏️ [PREENCHER — …]` nas células de discussão. Leia as saídas
executadas (tabelas e prints) e escreva um `textos.json` `{"MARCADOR": "markdown"}` para os 14
marcadores (lista em `DICAS` de `gerar_notebook.py`), seguindo `references/analise.md`.
Toda afirmação deve citar o número que a sustenta. Depois gere e execute de novo com
`--textos textos.json` e confira que nenhum `[PREENCHER` sobrou (`grep -c PREENCHER`).

Como os resultados são determinísticos (SEED = 42), os números não mudam entre execuções.
Se mudar algo no código, revise os textos.

### 4. Relatório e apresentação
Com os números finais, escreva o relatório (`references/relatorio.md`) e o roteiro da
apresentação (`references/apresentacao.md`). Formato: siga o que o usuário pedir; se ele não
disser, faça o relatório como documento e a apresentação como deck, usando os tipos de
artefato disponíveis na sessão. Reaproveite os gráficos do notebook (salve com
`fig.savefig(...)` se precisar de imagens).

### 5. Entrega
Entregue o `.ipynb` executado (com saídas), o CSV usado e, se pedirem, um `.py` equivalente
(`jupyter nbconvert --to script`). Lembre a equipe do prazo e de ensaiar a explicação do código.

## Decisões de modelagem já tomadas no template (e por quê)
- **Split 80/20 estratificado por `quality`**, mesmo split para regressão e classificação.
- **Pré-processamento:** mediana → winsorização 1–99% (`LimitadorQuantis`) → `StandardScaler`;
  `type` com `OneHotEncoder(drop="if_binary")`.
- **Regressão:** baseline (média), Linear, Ridge, Random Forest, HistGradientBoosting;
  GridSearch em Ridge, RF e HGB; coeficientes padronizados + importância por permutação.
- **Classificação:** esquemas A (0–4|5–6|7–10), B (0–5|6|7–10) e C (0–5|6–10);
  aprofundamento no A (o mais desbalanceado, que exercita balanceamento) — trocar com `--esquema`;
  métrica principal **F1-macro** (+ acurácia balanceada); Logística, SVM, RF, HGB;
  comparação sem ajuste × `class_weight` × SMOTE; GridSearch com SMOTE como hiperparâmetro.

Se o usuário pedir mudanças (outro modelo, RandomizedSearchCV, outras faixas), altere o
template mantendo o padrão: comentário curto em cada bloco + célula Markdown explicando a escolha.
