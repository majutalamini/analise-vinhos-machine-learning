# REALIZAR A ANALISE

# Como interpretar os resultados (para escrever as células de discussão)

Regra geral: cada parágrafo cita o número que o sustenta, explica o que significa em
linguagem simples e, quando fizer sentido, o porquê (química do vinho ou comportamento do modelo).
Textos curtos (3–6 frases ou poucos bullets por marcador). Português do Brasil.

## Valores de referência (grupo 1, SEED 42) — use para checar se algo saiu muito diferente

- Após limpeza: 5.369 linhas (1.125 duplicadas removidas), 4.002 brancos e 1.367 tintos.
- `quality`: 6 = 43,6 %, 5 = 33,0 %, 7 = 16,1 %; notas 3, 4, 8, 9 somam ~7 %.
- Correlação com `quality`: alcohol +0,47; density −0,33; volatile acidity −0,27; chlorides −0,20.
- Regressão (teste): RF RMSE 0,69 / MAE 0,53 / R² 0,38; HGB R² 0,37; Linear R² 0,30; baseline RMSE 0,88.
- Classificação esquema A (teste): HGB F1-macro 0,58; recall alta 0,69, baixa 0,45, média 0,76.
- F1-macro CV por esquema (RF): A 0,56 · B 0,60 · C 0,75.

## Pontos que valem discutir (são o que o professor costuma perguntar)

**EDA**

- A nota é quase uma variável de 3 valores (5, 6, 7 ≈ 93 %): os extremos são raros, o que
  limita qualquer modelo nas pontas.
- Caudas longas em residual sugar, chlorides, SO2 → justifica winsorização e mediana.
- Colinearidade: density × alcohol (forte, negativa) e density × residual sugar; SO2 livre × total.
  Isso afeta a interpretação dos coeficientes lineares.
- Tintos e brancos são quimicamente diferentes (chlorides, SO2 total, volatile acidity); `type`
  precisa entrar como feature.

**Outliers** — não removemos linhas porque (1) muitos "outliers" pelo IQR são vinhos reais
(ex.: vinhos doces com muito açúcar), (2) removeria justamente notas raras. Os valores
_impossíveis_ (erros de digitação) já foram tratados na limpeza.

**Regressão**

- Comparar sempre com o baseline: R² ≈ 0,3–0,4 é o esperado nesse dataset (a nota é subjetiva,
  dada por degustadores; a química explica só parte dela).
- Árvores > lineares → relações não lineares/interações (ex.: efeito do álcool depende do tipo).
- Ridge ≈ Linear: com ~4 mil amostras e 12 features, regularização quase não muda nada.
- Resíduos: o modelo "puxa para o meio" — superestima notas 3–4 e subestima 8–9 (regressão à
  média, consequência da concentração em 5–6).
- Coeficientes padronizados: alcohol é o maior positivo; volatile acidity negativo (acidez
  volátil = aroma de vinagre). Sinais "estranhos" (ex.: residual sugar positivo e density
  negativo ao mesmo tempo) vêm da colinearidade — density é quase função de álcool e açúcar,
  então o modelo divide o efeito entre elas. Diga isso explicitamente.
- Permutação: confirma alcohol como a feature dominante, seguida de volatile acidity e SO2 livre.

**Classificação**

- Esquema A é muito desbalanceado (baixa ≈ 4 %); B é mais equilibrado; C (binário) dá os
  maiores números, mas é o problema mais fácil — números de esquemas diferentes **não** são
  diretamente comparáveis (nº de classes diferente).
- Acurácia engana: no A o baseline acerta ~77 % prevendo só "média" e tem F1-macro 0,29.
- `class_weight` e SMOTE melhoram muito o F1-macro em relação a "sem ajuste" — principalmente o
  recall das classes raras — às custas de precisão (mais falsos "alta"/"baixa").
- Confusões: quase todos os erros são entre classes **vizinhas** (média ↔ alta, média ↔ baixa);
  quase nunca baixa ↔ alta. As notas de fronteira (6 e 7 no A; 4 e 5) são as mais erradas,
  porque um vinho nota 6 e um nota 7 têm química muito parecida.
- Feature decisiva: alcohol de novo, depois volatile acidity — coerente com a regressão.

**Conclusões / limitações**

- Nota sensorial é subjetiva e tem ruído; dataset não tem uva, safra, preço.
- Próximos passos possíveis: modelos separados por tipo, regressão ordinal, ajustar limiar de
  decisão por classe, mais dados das notas extremas.
