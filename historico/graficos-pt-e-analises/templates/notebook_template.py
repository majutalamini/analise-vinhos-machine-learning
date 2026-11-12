# %% [markdown]
# # Trabalho Final de Machine Learning — Predição e Classificação da Qualidade de Vinhos
#
# **Equipe:** {{EQUIPE}}
#
# **Dataset:** `{{CSV}}` — dados físico-químicos de vinhos tintos e brancos com nota de qualidade sensorial (0 a 10).
#
# **Objetivo:** construir um pipeline completo de Machine Learning em duas partes:
# 1. **Regressão** — prever a nota `quality` como valor contínuo;
# 2. **Classificação** — agrupar as notas em faixas e prever a faixa.
#
# Roteiro do notebook:
# 0. Configuração e bibliotecas
# 1. Carregamento e limpeza dos dados
# 2. Análise exploratória (histogramas, scatter plots, correlação)
# 3. Pré-processamento (ausentes, outliers, padronização)
# 4. Parte 1 — Regressão
# 5. Parte 2 — Classificação
# 6. Conclusões

# %% [markdown]
# ## 0. Configuração
# Bibliotecas usadas: `pandas`/`numpy` (dados), `matplotlib`/`seaborn` (gráficos),
# `scikit-learn` (modelos e métricas) e `imbalanced-learn` (SMOTE para balanceamento).
#
# Se faltar alguma: `pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn`
#
# ### Glossário rápido
#
# **Variáveis do vinho** (nomes originais do arquivo → significado):
#
# | Variável | Em português | O que mede |
# |---|---|---|
# | `fixed acidity` | acidez fixa | ácidos não voláteis (ex.: tartárico), dão "estrutura" ao vinho |
# | `volatile acidity` | acidez volátil | ácido acético; em excesso dá gosto de vinagre |
# | `citric acid` | ácido cítrico | acrescenta frescor |
# | `residual sugar` | açúcar residual | açúcar que sobra após a fermentação (g/L) |
# | `chlorides` | cloretos | quantidade de sal |
# | `free sulfur dioxide` | SO2 livre | dióxido de enxofre ativo, protege contra oxidação |
# | `total sulfur dioxide` | SO2 total | SO2 livre + combinado |
# | `density` | densidade | depende do álcool e do açúcar |
# | `pH` | pH | acidez geral (menor = mais ácido) |
# | `sulphates` | sulfatos | aditivo que atua como conservante |
# | `alcohol` | teor alcoólico | % de álcool por volume |
# | `type` | tipo | tinto (`red`) ou branco (`white`) |
# | `quality` | qualidade | **alvo**: nota sensorial de 0 a 10 |
#
# **Métricas e termos usados:**
# - **RMSE** — erro médio "quadrático", na mesma unidade da nota (quanto menor, melhor); pune erros grandes.
# - **MAE** — erro médio absoluto: em média, quantos pontos de nota o modelo erra.
# - **R²** — fração da variação da nota explicada pelo modelo (0 = não explica nada, 1 = explica tudo).
# - **F1-macro** — média do F1 de cada classe com peso igual; boa métrica quando as classes são desbalanceadas.
# - **Validação cruzada (CV)** — o treino é dividido em 5 partes; o modelo treina em 4 e é avaliado na 5ª, em rodízio.
# - **Winsorização** — limitar valores extremos aos percentis 1% e 99% em vez de apagar as linhas.
# - **SMOTE** — cria exemplos sintéticos das classes raras para equilibrar o treino.
# - **Data leakage (vazamento)** — quando informação do teste "vaza" para o treino e infla os resultados.

# %%
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (ConfusionMatrixDisplay, balanced_accuracy_score, classification_report,
                             f1_score, mean_absolute_error, r2_score, root_mean_squared_error)
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100

SEED = 42                       # semente fixa -> resultados reprodutíveis
CAMINHO_CSV = "{{CSV}}"         # ajuste se o arquivo estiver em outra pasta

# Nomes em português usados apenas nos gráficos e tabelas de apresentação
# (o código e o modelo continuam usando os nomes originais das colunas).
NOMES_PT = {
    "fixed acidity": "acidez fixa", "volatile acidity": "acidez volátil",
    "citric acid": "ácido cítrico", "residual sugar": "açúcar residual",
    "chlorides": "cloretos", "free sulfur dioxide": "SO2 livre",
    "total sulfur dioxide": "SO2 total", "density": "densidade", "pH": "pH",
    "sulphates": "sulfatos", "alcohol": "teor alcoólico", "quality": "qualidade",
    "type": "tipo", "type_white": "tipo = branco",
}
TIPO_PT = {"red": "Tinto", "white": "Branco"}
CORES_TIPO = {"Tinto": "#8c2d4a", "Branco": "#d9c27a"}

# %% [markdown]
# ## 1. Carregamento e limpeza dos dados
#
# Antes de qualquer análise, inspecionamos o arquivo bruto. Ao ler o CSV percebemos que
# **ele não está limpo**: algumas colunas que deveriam ser numéricas vieram como texto e
# vários valores estão em escala errada.

# %%
bruto = pd.read_csv(CAMINHO_CSV)
print(bruto.shape)
print(bruto.dtypes)
bruto.describe().T

# %%
# Exemplos de problemas encontrados no arquivo bruto
print("type:", bruto["type"].value_counts(dropna=False).to_dict())
print("quality:", bruto["quality"].value_counts(dropna=False).sort_index().to_dict())
alc = pd.to_numeric(bruto["alcohol"], errors="coerce")
print("alcohol não numérico (amostra):", bruto.loc[alc.isna() & bruto["alcohol"].notna(), "alcohol"].head(5).tolist())
print("chlorides acima de 1 (amostra):", bruto.loc[bruto["chlorides"] > 1, "chlorides"].head(5).tolist())
print("density acima de 1.04 (amostra):", bruto.loc[bruto["density"] > 1.04, "density"].head(5).tolist())
print("linhas duplicadas:", bruto.duplicated().sum())

# %% [markdown]
# ### Diagnóstico
# {{DIAGNOSTICO}}
#
# ### Estratégia de limpeza
# - **Vírgula decimal perdida** (ex.: `chlorides` 76 → 0,076; `density` 9.978 → 0,9978): cada variável
#   tem uma faixa plausível (baseada no dataset original da UCI). Valores fora dela são divididos por
#   potências de 10 até caírem na faixa; se nunca caem, viram `NaN` e são imputados depois.
# - **`alcohol` como texto** (ex.: `100.333.333.333.333` → 10,03): removemos os separadores e
#   reposicionamos a vírgula.
# - **`type` corrompido** (`whitewhite...`) → `white`; `type` ausente → linha removida.
# - **`quality` ausente ou fora de 0–10** → linha removida (o alvo não pode ser imputado).
# - **Duplicadas** → removidas, para que a mesma amostra não apareça em treino e teste
#   (isso inflaria artificialmente as métricas).
# - **Valores ausentes restantes** → imputados pela mediana **dentro do pipeline**, usando apenas
#   o conjunto de treino (evita *data leakage*).

# %%
FAIXAS = {
    "fixed acidity": (3.5, 16.5), "volatile acidity": (0.05, 1.8), "citric acid": (0.0, 1.7),
    "residual sugar": (0.5, 70.0), "chlorides": (0.005, 0.65), "free sulfur dioxide": (1.0, 300.0),
    "total sulfur dioxide": (5.0, 450.0), "density": (0.985, 1.04), "pH": (2.7, 4.1),
    "sulphates": (0.2, 2.1), "alcohol": (8.0, 15.0),
}
# Valores originais com 3 casas decimais que perderam "0," -> inteiro / 1000
DIVISOR_INTEIROS = {"chlorides": 1000, "volatile acidity": 1000}


def reescalar(valor, minimo, maximo, divisor_inteiro=None):
    """Corrige valores que perderam a vírgula decimal; devolve NaN se irrecuperável."""
    if pd.isna(valor):
        return np.nan
    if minimo <= valor <= maximo:
        return valor
    if divisor_inteiro and float(valor).is_integer():
        v = valor / divisor_inteiro
        return v if minimo <= v <= maximo else np.nan
    v = float(valor)
    for _ in range(6):
        v /= 10
        if minimo <= v <= maximo:
            return v
    return np.nan


def corrigir_alcohol(texto):
    """'100.333.333.333' -> 10.0333 ; '9.39.39.39' -> 9.3 ; '10.5' -> 10.5"""
    if pd.isna(texto):
        return np.nan
    s = str(texto).strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        pass
    partes = s.split(".")
    if len(partes) > 2 and len(set(partes[1:-1])) == 1 and len(partes[1]) == 2:
        return float(f"{partes[0]}.{partes[1][0]}")
    digitos = re.sub(r"\D", "", s)
    for k in (1, 2):
        try:
            v = float(digitos[:k] + "." + digitos[k:])
        except ValueError:
            continue
        if FAIXAS["alcohol"][0] <= v <= FAIXAS["alcohol"][1]:
            return v
    return np.nan


def corrigir_tipo(texto):
    if pd.isna(texto):
        return np.nan
    s = str(texto).strip().lower()
    return "white" if s.startswith("white") else "red" if s.startswith("red") else np.nan


df = bruto.copy()
log = {}
df["alcohol"] = df["alcohol"].apply(corrigir_alcohol)
for col, (lo, hi) in FAIXAS.items():
    df[col] = pd.to_numeric(df[col], errors="coerce")
    fora = df[col].notna() & ~df[col].between(lo, hi)
    df[col] = df[col].apply(lambda v: reescalar(v, lo, hi, DIVISOR_INTEIROS.get(col)))
    log[col] = {"fora_da_faixa": int(fora.sum()), "virou_NaN": int((fora & df[col].isna()).sum())}
df["type"] = df["type"].apply(corrigir_tipo)

df["quality"] = pd.to_numeric(df["quality"], errors="coerce")
invalida = df["quality"].isna() | ~df["quality"].between(0, 10) | df["type"].isna()
print("linhas removidas (quality/type inválidos):", int(invalida.sum()))
df = df[~invalida].copy()
df["quality"] = df["quality"].astype(int)

n_dup = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
print("duplicadas removidas:", n_dup)
print("linhas finais:", len(df))
pd.DataFrame(log).T

# %%
# Conferência: depois da limpeza as estatísticas ficam coerentes com a literatura
print("Valores ausentes restantes (serão imputados no pipeline):")
print(df.isna().sum()[df.isna().sum() > 0])
df.describe().T.round(3)

# %% [markdown]
# ## 2. Análise exploratória (EDA)
# ### 2.1 Distribuição do alvo `quality`

# %%
# Coluna auxiliar só para os gráficos (rótulos em português); não entra no modelo.
df["tipo_pt"] = df["type"].map(TIPO_PT)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(x="quality", data=df, ax=ax[0], color="#8c2d4a")
ax[0].set_title("Distribuição da nota de qualidade")
ax[0].set_xlabel("Nota de qualidade (0 a 10)"); ax[0].set_ylabel("Quantidade de vinhos")
sns.countplot(x="quality", hue="tipo_pt", data=df, ax=ax[1], palette=CORES_TIPO)
ax[1].set_title("Nota de qualidade por tipo de vinho")
ax[1].set_xlabel("Nota de qualidade (0 a 10)"); ax[1].set_ylabel("Quantidade de vinhos")
ax[1].legend(title="Tipo de vinho")
plt.tight_layout(); plt.show()
print(df["quality"].value_counts(normalize=True).sort_index().round(3))

# %% [markdown]
# {{EDA_ALVO}}
#
# ### 2.2 Histogramas das variáveis físico-químicas

# %%
FEATURES_NUM = list(FAIXAS.keys())
eixos = df[FEATURES_NUM].rename(columns=NOMES_PT).hist(bins=40, figsize=(16, 11),
                                                       color="#4a6fa5", edgecolor="white")
for ax in eixos.flat:
    ax.set_xlabel("valor medido"); ax.set_ylabel("nº de vinhos")
plt.suptitle("Histogramas das variáveis físico-químicas (quantos vinhos há em cada faixa de valor)", y=1.0)
plt.tight_layout(); plt.show()

# %% [markdown]
# {{EDA_HIST}}
#
# ### 2.3 Boxplots — detecção visual de outliers

# %%
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for ax, col in zip(axes.flat, FEATURES_NUM):
    sns.boxplot(y=df[col], ax=ax, color="#a5c8e4")
    ax.set_title(NOMES_PT[col]); ax.set_ylabel("valor medido")
axes.flat[-1].axis("off")
# Legenda de leitura no espaço vazio: caixa = 50% central dos vinhos; pontos = outliers
axes.flat[-1].text(0.0, 0.5, "Como ler o boxplot:\n• linha no meio da caixa = mediana\n"
                   "• caixa = 50% centrais dos vinhos\n• hastes = faixa 'normal'\n"
                   "• pontos isolados = outliers", fontsize=10, va="center")
plt.suptitle("Boxplots das variáveis físico-químicas", y=1.0)
plt.tight_layout(); plt.show()

# Contagem de outliers pelo critério IQR (1,5 × intervalo interquartil)
q1, q3 = df[FEATURES_NUM].quantile(0.25), df[FEATURES_NUM].quantile(0.75)
iqr = q3 - q1
outliers_iqr = ((df[FEATURES_NUM] < q1 - 1.5 * iqr) | (df[FEATURES_NUM] > q3 + 1.5 * iqr)).sum()
outliers_iqr.rename(index=NOMES_PT).sort_values(ascending=False).to_frame("nº de outliers (IQR)")

# %% [markdown]
# {{EDA_OUTLIERS}}
#
# ### 2.4 Matriz de correlação

# %%
corr = df[FEATURES_NUM + ["quality"]].corr()
plt.figure(figsize=(11, 9))
sns.heatmap(corr.rename(index=NOMES_PT, columns=NOMES_PT), annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, cbar_kws={"shrink": .8, "label": "correlação (−1 a +1)"})
plt.title("Matriz de correlação de Pearson\n(azul = sobem juntas · vermelho = uma sobe e a outra desce)")
plt.show()
corr["quality"].drop("quality").rename(index=NOMES_PT).sort_values(key=abs, ascending=False).to_frame("correlação com a qualidade")

# %% [markdown]
# {{EDA_CORR}}
#
# ### 2.5 Scatter plots das variáveis mais relacionadas à qualidade

# %%
top4 = corr["quality"].drop("quality").abs().sort_values(ascending=False).index[:4]
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, col in zip(axes, top4):
    sns.stripplot(x="quality", y=col, data=df, ax=ax, alpha=.25, jitter=.3, size=2.5, color="#4a6fa5")
    sns.pointplot(x="quality", y=col, data=df, ax=ax, color="#c0392b", errorbar=None)
    ax.set_title(f"{NOMES_PT[col]} × qualidade")
    ax.set_xlabel("Nota de qualidade"); ax.set_ylabel(NOMES_PT[col])
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color="#4a6fa5", alpha=.6, label="cada vinho"),
                       Line2D([], [], color="#c0392b", marker="o", label="média da nota")],
              fontsize=8, loc="best")
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
sns.scatterplot(x="alcohol", y="density", hue="quality", data=df, palette="viridis", s=12, ax=ax[0])
ax[0].set_title("Teor alcoólico × densidade (cor = nota de qualidade)")
ax[0].set_xlabel("Teor alcoólico (% vol.)"); ax[0].set_ylabel("Densidade (g/cm³)")
ax[0].legend(title="Qualidade", fontsize=8)
sns.scatterplot(x="total sulfur dioxide", y="chlorides", hue="tipo_pt", data=df, s=12, ax=ax[1],
                palette=CORES_TIPO)
ax[1].set_title("Tintos e brancos separam bem em SO2 total × cloretos")
ax[1].set_xlabel("SO2 total (mg/L)"); ax[1].set_ylabel("Cloretos (g/L)")
ax[1].legend(title="Tipo de vinho")
plt.tight_layout(); plt.show()

# %% [markdown]
# {{EDA_SCATTER}}
#
# ## 3. Pré-processamento
#
# Decisões (todas aplicadas **dentro de um `Pipeline`**, ajustadas só no treino):
# 1. **Ausentes** → `SimpleImputer(strategy="median")` (mediana é robusta a outliers).
# 2. **Outliers** → não removemos linhas (perderíamos vinhos reais e raros, inclusive das notas extremas);
#    em vez disso, **limitamos (winsorização)** cada variável aos percentis 1% e 99% do treino.
# 3. **Padronização** → `StandardScaler` (média 0, desvio 1). Essencial para modelos lineares, SVM e KNN e
#    permite comparar coeficientes entre variáveis.
# 4. **`type`** → `OneHotEncoder` (vira a variável binária `type_white`).

# %%
class LimitadorQuantis(BaseEstimator, TransformerMixin):
    """Winsorização: corta valores abaixo/acima dos quantis aprendidos no treino."""

    def __init__(self, inferior=0.01, superior=0.99):
        self.inferior = inferior
        self.superior = superior

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lim_inf_ = np.nanquantile(X, self.inferior, axis=0)
        self.lim_sup_ = np.nanquantile(X, self.superior, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lim_inf_, self.lim_sup_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def criar_preprocessador():
    numerico = Pipeline([
        ("imputacao", SimpleImputer(strategy="median")),
        ("outliers", LimitadorQuantis(0.01, 0.99)),
        ("padronizacao", StandardScaler()),
    ])
    categorico = OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([
        ("num", numerico, FEATURES_NUM),
        ("cat", categorico, ["type"]),
    ], verbose_feature_names_out=False)


X = df[FEATURES_NUM + ["type"]]
y = df["quality"]

# Divisão treino/teste estratificada pela nota -> mesmas proporções nos dois conjuntos.
# O conjunto de teste só é usado no final, para a avaliação definitiva.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)
print("treino:", X_train.shape, "| teste:", X_test.shape)

# %% [markdown]
# ## 4. Parte 1 — Regressão (predição da nota)
#
# ### 4.1 Modelos e validação cruzada
# Comparamos um **baseline** (prevê sempre a média) com quatro modelos:
# - **Regressão Linear** — simples e interpretável (coeficientes);
# - **Ridge** — linear com regularização L2 (lida melhor com variáveis correlacionadas, como density × alcohol);
# - **Random Forest** — conjunto de árvores, captura não linearidades e interações;
# - **HistGradientBoosting** — boosting de árvores, geralmente o mais forte em dados tabulares.
#
# Métricas: **RMSE** (penaliza erros grandes, na unidade da nota), **MAE** (erro médio absoluto) e
# **R²** (fração da variância explicada). Validação cruzada com 5 folds no treino.

# %%
modelos_reg = {
    "Baseline (média)": DummyRegressor(strategy="mean"),
    "Regressão Linear": LinearRegression(),
    "Ridge": Ridge(random_state=SEED),
    "Random Forest": RandomForestRegressor(random_state=SEED, n_jobs=-1),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=SEED),
}
cv_reg = KFold(n_splits=5, shuffle=True, random_state=SEED)
metricas_reg = {"RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error", "R2": "r2"}

linhas = []
for nome, modelo in modelos_reg.items():
    pipe = Pipeline([("prep", criar_preprocessador()), ("modelo", modelo)])
    res = cross_validate(pipe, X_train, y_train, cv=cv_reg, scoring=metricas_reg, n_jobs=-1)
    linhas.append({"Modelo": nome,
                   "RMSE": -res["test_RMSE"].mean(), "MAE": -res["test_MAE"].mean(),
                   "R²": res["test_R2"].mean(), "R² (desvio)": res["test_R2"].std()})
resultado_cv_reg = pd.DataFrame(linhas).set_index("Modelo").round(4)
resultado_cv_reg

# %% [markdown]
# {{REG_CV}}
#
# ### 4.2 Ajuste de hiperparâmetros (GridSearchCV)
# Otimizamos o RMSE (via `neg_root_mean_squared_error`) com validação cruzada de 5 folds.

# %%
grades_reg = {
    "Ridge": (Ridge(random_state=SEED),
              {"modelo__alpha": [0.01, 0.1, 1, 10, 50, 100]}),
    "Random Forest": (RandomForestRegressor(random_state=SEED, n_jobs=-1),
                      {"modelo__n_estimators": [200, 400],
                       "modelo__max_depth": [None, 20],
                       "modelo__min_samples_leaf": [1, 3],
                       "modelo__max_features": ["sqrt", 0.5]}),
    "HistGradientBoosting": (HistGradientBoostingRegressor(random_state=SEED),
                             {"modelo__learning_rate": [0.03, 0.1],
                              "modelo__max_iter": [200, 500],
                              "modelo__max_leaf_nodes": [15, 31],
                              "modelo__l2_regularization": [0.0, 1.0]}),
}

melhores_reg = {}
for nome, (modelo, grade) in grades_reg.items():
    busca = GridSearchCV(Pipeline([("prep", criar_preprocessador()), ("modelo", modelo)]),
                         grade, cv=cv_reg, scoring="neg_root_mean_squared_error", n_jobs=-1)
    busca.fit(X_train, y_train)
    melhores_reg[nome] = busca.best_estimator_
    print(f"{nome:22s} RMSE CV = {-busca.best_score_:.4f}  |  {busca.best_params_}")

# %% [markdown]
# ### 4.3 Avaliação final no conjunto de teste

# %%
melhores_reg["Regressão Linear"] = Pipeline([("prep", criar_preprocessador()),
                                             ("modelo", LinearRegression())]).fit(X_train, y_train)
linhas = []
for nome, pipe in melhores_reg.items():
    pred = pipe.predict(X_test)
    linhas.append({"Modelo": nome, "RMSE": root_mean_squared_error(y_test, pred),
                   "MAE": mean_absolute_error(y_test, pred), "R²": r2_score(y_test, pred),
                   "Acerto ao arredondar": (np.clip(np.round(pred), 3, 9) == y_test).mean()})
resultado_teste_reg = pd.DataFrame(linhas).set_index("Modelo").sort_values("RMSE").round(4)
resultado_teste_reg

# %%
melhor_nome_reg = resultado_teste_reg.index[0]
melhor_reg = melhores_reg[melhor_nome_reg]
pred = melhor_reg.predict(X_test)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
sns.boxplot(x=y_test, y=pred, ax=ax[0], color="#a5c8e4")
notas = np.sort(y_test.unique())
ax[0].plot(range(len(notas)), notas, "r--", label="previsão perfeita")
ax[0].set_xlabel("Nota real"); ax[0].set_ylabel("Nota prevista pelo modelo")
ax[0].set_title(f"{melhor_nome_reg}: nota prevista × nota real"); ax[0].legend()
sns.histplot(y_test - pred, bins=40, ax=ax[1], color="#4a6fa5")
ax[1].axvline(0, color="r", ls="--", label="erro zero (previsão exata)")
ax[1].set_title("Distribuição dos resíduos (nota real − nota prevista)")
ax[1].set_xlabel("Resíduo (em pontos de nota)"); ax[1].set_ylabel("Quantidade de vinhos")
ax[1].legend()
plt.tight_layout(); plt.show()

# %% [markdown]
# {{REG_TESTE}}
#
# ### 4.4 Interpretação — quais variáveis mais impactam a nota?
# **Coeficientes da Regressão Linear:** como as variáveis foram padronizadas, cada coeficiente indica
# quanto a nota muda quando a variável aumenta **1 desvio-padrão**, mantendo as demais constantes.

# %%
lin = melhores_reg["Regressão Linear"]
coef = pd.Series(lin.named_steps["modelo"].coef_,
                 index=lin.named_steps["prep"].get_feature_names_out()).sort_values()
ridge = melhores_reg["Ridge"]
coef_ridge = pd.Series(ridge.named_steps["modelo"].coef_,
                       index=ridge.named_steps["prep"].get_feature_names_out())

fig, ax = plt.subplots(figsize=(9, 5))
coef.rename(index=NOMES_PT).plot.barh(ax=ax, color=["#c0392b" if c < 0 else "#27ae60" for c in coef])
ax.set_title("Coeficientes padronizados — Regressão Linear"); ax.axvline(0, color="k", lw=.8)
ax.set_xlabel("Variação na nota para +1 desvio-padrão da variável (demais constantes)")
ax.set_ylabel("Variável")
ax.legend(handles=[Line2D([], [], color="#27ae60", lw=6, label="aumenta a nota"),
                   Line2D([], [], color="#c0392b", lw=6, label="diminui a nota")], loc="lower right")
plt.tight_layout(); plt.show()
pd.DataFrame({"Linear": coef, "Ridge": coef_ridge}).rename(index=NOMES_PT).round(3)

# %%
# Importância por permutação no teste: quanto o R² cai ao embaralhar cada variável.
# Funciona para qualquer modelo e é mais confiável que a importância "impurity" das árvores.
imp = permutation_importance(melhor_reg, X_test, y_test, n_repeats=10, random_state=SEED,
                             scoring="r2", n_jobs=-1)
imp_reg = pd.Series(imp.importances_mean, index=X_test.columns).rename(index=NOMES_PT).sort_values()
imp_reg.plot.barh(figsize=(8, 5), color="#4a6fa5",
                  title=f"Importância por permutação — {melhor_nome_reg}")
plt.xlabel("queda no R² ao embaralhar a variável (maior = mais importante)")
plt.ylabel("Variável"); plt.tight_layout(); plt.show()
imp_reg.sort_values(ascending=False).round(4).to_frame("queda no R²")

# %% [markdown]
# ### 4.5 Discussão da regressão
# {{REG_DISCUSSAO}}
#
# ## 5. Parte 2 — Classificação (faixas de qualidade)
#
# ### 5.1 Criação da variável alvo categórica — testando diferentes faixas
# O enunciado pede para testar diferentes *ranges*. Testamos três discretizações:
#
# | Esquema | Faixas | Ideia |
# |---|---|---|
# | **A** | baixa (0–4) · média (5–6) · alta (7–10) | faixas "naturais" da escala sensorial |
# | **B** | baixa (0–5) · média (6) · alta (7–10) | divide a massa central, classes menos desiguais |
# | **C** | ruim (0–5) · bom (6–10) | binário, o mais equilibrado |

# %%
ESQUEMAS = {
    "A: 0-4 | 5-6 | 7-10": ([0, 4, 6, 10], ["baixa", "média", "alta"]),
    "B: 0-5 | 6 | 7-10":   ([0, 5, 6, 10], ["baixa", "média", "alta"]),
    "C: 0-5 | 6-10":       ([0, 5, 10], ["ruim", "bom"]),
}


def discretizar(notas, esquema):
    limites, rotulos = ESQUEMAS[esquema]
    return pd.cut(notas, bins=limites, labels=rotulos, include_lowest=True).astype(str)


fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, esquema in zip(axes, ESQUEMAS):
    dist = discretizar(df["quality"], esquema).value_counts(normalize=True)[ESQUEMAS[esquema][1]]
    dist.plot.bar(ax=ax, color="#8c2d4a", rot=0)
    ax.set_title(f"Esquema {esquema}"); ax.set_ylabel("proporção de vinhos")
    ax.set_xlabel("faixa de qualidade")
    for i, v in enumerate(dist):
        ax.text(i, v + .01, f"{v:.1%}", ha="center")
plt.tight_layout(); plt.show()

# %% [markdown]
# {{CLF_DISTRIBUICAO}}
#
# ### 5.2 Escolha da métrica
# Como as classes são **desbalanceadas**, a acurácia engana: um modelo que só prevê "média" já
# acerta a maioria. Usamos o **F1-macro** como métrica principal — a média do F1 de cada classe, com
# **peso igual** para todas — e a **acurácia balanceada** (média do recall por classe) como apoio.
# Assim, errar todos os vinhos de qualidade baixa ou alta é punido de verdade.
#
# ### 5.3 Comparação inicial: 4 classificadores × 3 esquemas
# Classificadores: **Regressão Logística**, **SVM (RBF)**, **Random Forest** e **HistGradientBoosting**,
# todos com `class_weight="balanced"` (dá mais peso às classes raras) e um baseline que sempre
# prevê a classe majoritária. Os rótulos de texto são convertidos em números com `LabelEncoder`.

# %%
def classificadores():
    return {
        "Baseline (majoritária)": DummyClassifier(strategy="most_frequent"),
        "Regressão Logística": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "SVM (RBF)": SVC(class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
    }


cv_clf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
linhas = []
for esquema in ESQUEMAS:
    # mesmo split do treino/teste da regressão (mesmos índices)
    y_tr = LabelEncoder().fit(ESQUEMAS[esquema][1]).transform(discretizar(y_train, esquema))
    for nome, modelo in classificadores().items():
        pipe = Pipeline([("prep", criar_preprocessador()), ("modelo", modelo)])
        res = cross_validate(pipe, X_train, y_tr, cv=cv_clf, n_jobs=-1,
                             scoring={"f1": "f1_macro", "bal": "balanced_accuracy", "acc": "accuracy"})
        linhas.append({"Esquema": esquema, "Modelo": nome, "F1-macro": res["test_f1"].mean(),
                       "Acur. balanceada": res["test_bal"].mean(), "Acurácia": res["test_acc"].mean()})
comparacao = pd.DataFrame(linhas).round(4)
comparacao.pivot(index="Modelo", columns="Esquema", values="F1-macro")

# %%
comparacao.set_index(["Esquema", "Modelo"])

# %% [markdown]
# {{CLF_COMPARACAO}}
#
# ### 5.4 Esquema principal e balanceamento de classes
# Aprofundamos no esquema **{{ESQUEMA_PRINCIPAL}}**. Comparamos três estratégias para lidar com o
# desbalanceamento, usando o Random Forest:
# 1. **Nenhum ajuste**;
# 2. **`class_weight="balanced"`** — pesos inversamente proporcionais à frequência da classe;
# 3. **SMOTE** (oversampling sintético) — cria exemplos artificiais das classes raras **apenas nos folds
#    de treino** (por isso usamos o `Pipeline` do `imblearn`; aplicar SMOTE antes do split vazaria dados).

# %%
ESQ = "{{ESQUEMA_PRINCIPAL}}"
codificador = LabelEncoder().fit(ESQUEMAS[ESQ][1])
y_train_c = codificador.transform(discretizar(y_train, ESQ))
y_test_c = codificador.transform(discretizar(y_test, ESQ))
# LabelEncoder numera as classes em ordem alfabética
print("Classes codificadas:", {str(c): int(i) for c, i in zip(codificador.classes_,
                                                            codificador.transform(codificador.classes_))})
print("Distribuição no treino:", pd.Series(codificador.inverse_transform(y_train_c)).value_counts().to_dict())

estrategias = {
    "Sem ajuste": ImbPipeline([("prep", criar_preprocessador()),
                               ("modelo", RandomForestClassifier(random_state=SEED, n_jobs=-1))]),
    "class_weight=balanced": ImbPipeline([("prep", criar_preprocessador()),
                                          ("modelo", RandomForestClassifier(class_weight="balanced",
                                                                            random_state=SEED, n_jobs=-1))]),
    "SMOTE": ImbPipeline([("prep", criar_preprocessador()), ("smote", SMOTE(random_state=SEED)),
                          ("modelo", RandomForestClassifier(random_state=SEED, n_jobs=-1))]),
}
linhas = []
for nome, pipe in estrategias.items():
    res = cross_validate(pipe, X_train, y_train_c, cv=cv_clf, n_jobs=-1,
                         scoring={"f1": "f1_macro", "bal": "balanced_accuracy"})
    linhas.append({"Estratégia": nome, "F1-macro": res["test_f1"].mean(),
                   "Acur. balanceada": res["test_bal"].mean()})
pd.DataFrame(linhas).set_index("Estratégia").round(4)

# %% [markdown]
# {{CLF_BALANCEAMENTO}}
#
# ### 5.5 Ajuste de hiperparâmetros (GridSearchCV, F1-macro)

# %%
grades_clf = {
    "Regressão Logística": (
        ImbPipeline([("prep", criar_preprocessador()), ("smote", SMOTE(random_state=SEED)),
                     ("modelo", LogisticRegression(max_iter=3000))]),
        {"smote": [SMOTE(random_state=SEED), "passthrough"],
         "modelo__C": [0.01, 0.1, 1, 10],
         "modelo__class_weight": [None, "balanced"]}),
    "SVM (RBF)": (
        ImbPipeline([("prep", criar_preprocessador()), ("smote", "passthrough"),
                     ("modelo", SVC(class_weight="balanced", random_state=SEED))]),
        {"modelo__C": [1, 10, 30], "modelo__gamma": ["scale", 0.1, 0.3]}),
    "Random Forest": (
        ImbPipeline([("prep", criar_preprocessador()), ("smote", "passthrough"),
                     ("modelo", RandomForestClassifier(random_state=SEED, n_jobs=-1))]),
        {"smote": [SMOTE(random_state=SEED), "passthrough"],
         "modelo__class_weight": [None, "balanced_subsample"],
         "modelo__n_estimators": [300],
         "modelo__max_depth": [None, 15],
         "modelo__min_samples_leaf": [1, 3]}),
    "HistGradientBoosting": (
        ImbPipeline([("prep", criar_preprocessador()), ("smote", "passthrough"),
                     ("modelo", HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED))]),
        {"modelo__learning_rate": [0.05, 0.1],
         "modelo__max_iter": [200, 400],
         "modelo__max_leaf_nodes": [15, 31]}),
}

melhores_clf, linhas = {}, []
for nome, (pipe, grade) in grades_clf.items():
    busca = GridSearchCV(pipe, grade, cv=cv_clf, scoring="f1_macro", n_jobs=-1)
    busca.fit(X_train, y_train_c)
    melhores_clf[nome] = busca.best_estimator_
    print(f"{nome:22s} F1-macro CV = {busca.best_score_:.4f}  |  {busca.best_params_}")

# %% [markdown]
# ### 5.6 Avaliação final no conjunto de teste

# %%
linhas = []
for nome, pipe in melhores_clf.items():
    pred_c = pipe.predict(X_test)
    linhas.append({"Modelo": nome, "F1-macro": f1_score(y_test_c, pred_c, average="macro"),
                   "Acur. balanceada": balanced_accuracy_score(y_test_c, pred_c),
                   "Acurácia": (pred_c == y_test_c).mean()})
resultado_teste_clf = pd.DataFrame(linhas).set_index("Modelo").sort_values("F1-macro", ascending=False).round(4)
resultado_teste_clf

# %%
melhor_nome_clf = resultado_teste_clf.index[0]
melhor_clf = melhores_clf[melhor_nome_clf]
pred_c = melhor_clf.predict(X_test)
print(f"Melhor modelo: {melhor_nome_clf}\n")
print(classification_report(y_test_c, pred_c, target_names=codificador.classes_, digits=3))

ordem = ESQUEMAS[ESQ][1]
idx = codificador.transform(ordem)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ConfusionMatrixDisplay.from_predictions(y_test_c, pred_c, labels=idx, display_labels=ordem,
                                        cmap="Blues", ax=ax[0], colorbar=False)
ax[0].set_title("Matriz de confusão (nº de vinhos)")
ConfusionMatrixDisplay.from_predictions(y_test_c, pred_c, labels=idx, display_labels=ordem,
                                        normalize="true", values_format=".0%", cmap="Blues",
                                        ax=ax[1], colorbar=False)
ax[1].set_title("Normalizada por linha (% de acerto de cada classe real)")
for a in ax:   # linhas = classe real, colunas = classe prevista; diagonal = acertos
    a.set_xlabel("Faixa prevista pelo modelo"); a.set_ylabel("Faixa real do vinho")
plt.tight_layout(); plt.show()

# %%
# Matrizes de confusão de todos os modelos ajustados, para comparar os padrões de erro
fig, axes = plt.subplots(1, len(melhores_clf), figsize=(5 * len(melhores_clf), 4.5))
for ax, (nome, pipe) in zip(axes, melhores_clf.items()):
    ConfusionMatrixDisplay.from_predictions(y_test_c, pipe.predict(X_test), labels=idx,
                                            display_labels=ordem, normalize="true",
                                            values_format=".0%", cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(nome)
    ax.set_xlabel("Faixa prevista"); ax.set_ylabel("Faixa real")
plt.tight_layout(); plt.show()

# %%
# Quais notas originais são mais confundidas? (erros do melhor modelo por nota real)
erros = pd.DataFrame({"nota": y_test.values, "real": codificador.inverse_transform(y_test_c),
                      "previsto": codificador.inverse_transform(pred_c)})
erros["acertou"] = erros["real"] == erros["previsto"]
erros.groupby("nota")["acertou"].agg(["count", "mean"]).rename(columns={"count": "n", "mean": "taxa de acerto"}).round(3)

# %%
imp = permutation_importance(melhor_clf, X_test, y_test_c, n_repeats=10, random_state=SEED,
                             scoring="f1_macro", n_jobs=-1)
imp_clf = pd.Series(imp.importances_mean, index=X_test.columns).rename(index=NOMES_PT).sort_values()
imp_clf.plot.barh(figsize=(8, 5), color="#8c2d4a",
                  title=f"Importância por permutação — {melhor_nome_clf}")
plt.xlabel("queda no F1-macro ao embaralhar a variável (maior = mais importante)")
plt.ylabel("Variável"); plt.tight_layout(); plt.show()
imp_clf.sort_values(ascending=False).round(4).to_frame("queda no F1-macro")

# %% [markdown]
# ### 5.7 Discussão da classificação
# {{CLF_DISCUSSAO}}
#
# ## 6. Conclusões
# {{CONCLUSOES}}
