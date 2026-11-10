"""Análise e treinamento dos modelos de qualidade de vinhos do Grupo 1.

Execute com Python 3.12 e as dependências de requirements.txt.
Os gráficos são exibidos ao longo da execução.
"""


def main():
    # Trabalho Final de Machine Learning — Predição e Classificação da Qualidade de Vinhos

    # 0. Configuração

    import os
    from pathlib import Path

    # Paraleliza as dobras em threads, limitando as threads internas de cada ajuste.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                                  RandomForestClassifier, RandomForestRegressor)
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
    from sklearn.metrics import (ConfusionMatrixDisplay, balanced_accuracy_score, classification_report, confusion_matrix,
                                 f1_score, mean_absolute_error, r2_score, root_mean_squared_error)
    from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold, cross_validate,
                                         train_test_split)
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
    from sklearn.svm import SVC
    from sklearn.utils.validation import validate_data
    from imblearn.over_sampling import SMOTENC
    from imblearn.pipeline import Pipeline as ImbPipeline
    from joblib import parallel_config

    from IPython.display import display
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 100

    SEED = 42                       # semente fixa -> resultados reprodutíveis
    RAIZ_PROJETO = Path(__file__).resolve().parent
    CAMINHO_CSV = RAIZ_PROJETO / "dados/group_1_winequality.csv"
    N_JOBS = min(6, os.cpu_count() or 1)



    # 1. Carregamento e limpeza dos dados

    from limpeza import COLUNAS_NUMERICAS, ler_csv, limpar_dados

    bruto = ler_csv(CAMINHO_CSV)
    print(bruto.shape)
    print(bruto.dtypes)
    display(bruto.describe().T)



    # Exemplos de problemas encontrados no arquivo bruto
    print("type:", bruto["type"].value_counts(dropna=False).to_dict())
    print("quality:", bruto["quality"].value_counts(dropna=False).sort_index().to_dict())
    alc = pd.to_numeric(bruto["alcohol"], errors="coerce")
    print("alcohol não numérico (amostra):", bruto.loc[alc.isna() & bruto["alcohol"].notna(), "alcohol"].str.slice(0, 32).head(5).tolist())
    print("chlorides acima de 1 (amostra):", bruto.loc[bruto["chlorides"] > 1, "chlorides"].head(5).tolist())
    print("density acima de 1.04 (amostra):", bruto.loc[bruto["density"] > 1.04, "density"].head(5).tolist())
    print("linhas duplicadas:", bruto.duplicated().sum())



    # Diagnóstico

    # Estratégia de limpeza

    df, auditoria = limpar_dados(bruto, verbose=False)
    chaves_remocao = ["linhas_quality_invalida_removidas", "linhas_sem_type_removidas", "duplicadas_removidas"]
    assert len(bruto) - sum(auditoria[chave] for chave in chaves_remocao) == len(df)
    assert pd.api.types.is_integer_dtype(df["quality"]) and df["quality"].between(0, 10).all()
    assert df["type"].isin(["red", "white"]).all()
    assert not df.duplicated().any()

    print("Auditoria de linhas:", {k: v for k, v in auditoria.items() if isinstance(v, int)})
    tabela_limpeza = pd.DataFrame(auditoria["numericas"]).T
    tabela_limpeza["ausentes após limpeza"] = df[tabela_limpeza.index].isna().sum()
    display(tabela_limpeza)



    # Médias calculadas nos dois CSVs originais da UCI, DOI 10.24432/C56S3T.
    # A comparação é agregada: deduplicação e imputáveis podem mudar as médias do grupo.
    medias_uci = pd.DataFrame({
        "chlorides": {"red": 0.08746654, "white": 0.04577236},
        "volatile acidity": {"red": 0.52782051, "white": 0.27824112},
        "density": {"red": 0.99674668, "white": 0.99402738},
    })
    medias_uci.index.name = "type"
    medias_grupo = df.groupby("type")[medias_uci.columns].mean()
    comparacao_uci = pd.concat({"Grupo 1 (limpo)": medias_grupo, "UCI (bruto)": medias_uci}, axis=1)
    display(comparacao_uci.round(5))



    FEATURES_NUM = COLUNAS_NUMERICAS
    X, y = df[FEATURES_NUM + ["type"]], df["quality"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    assert X_train.index.intersection(X_test.index).empty
    assert len(X_train) + len(X_test) == len(df)
    perfis_treino = set(pd.util.hash_pandas_object(X_train, index=False))
    perfis_teste = set(pd.util.hash_pandas_object(X_test, index=False))
    print("Perfis físico-químicos iguais nos dois conjuntos:", len(perfis_treino & perfis_teste))
    treino = df.loc[X_train.index]
    print("Treino:", len(X_train), "| teste:", len(X_test))



    # 2. Análise exploratória (EDA)

    # 2.1 Distribuição do alvo `quality`

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(x="quality", data=treino, ax=ax[0], color="#8c2d4a")
    ax[0].set_title("Distribuição de quality")
    sns.countplot(x="quality", hue="type", data=treino, ax=ax[1], palette={"red": "#8c2d4a", "white": "#d9c27a"})
    ax[1].legend(title="Tipo", labels=["Tinto" if t.get_text() == "red" else "Branco" for t in ax[1].legend_.texts])
    ax[1].set_title("quality por tipo de vinho")
    plt.tight_layout()
    plt.show()
    print(treino["quality"].value_counts(normalize=True).sort_index().round(3))



    # 2.2 Histogramas das variáveis físico-químicas

    eixos_hist = treino[FEATURES_NUM].hist(bins=40, figsize=(16, 11), color="#4a6fa5", edgecolor="white")
    fig = eixos_hist.flat[0].figure
    plt.suptitle("Histogramas das variáveis numéricas", y=1.0)
    plt.tight_layout()
    plt.show()



    # 2.3 Boxplots — detecção visual de outliers

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for ax, col in zip(axes.flat, FEATURES_NUM):
        sns.boxplot(y=treino[col], ax=ax, color="#a5c8e4")
        ax.set_title(col)
    axes.flat[-1].axis("off")
    plt.tight_layout()
    plt.show()

    # Contagem de outliers pelo critério IQR (1,5 × intervalo interquartil)
    q1, q3 = treino[FEATURES_NUM].quantile(0.25), treino[FEATURES_NUM].quantile(0.75)
    iqr = q3 - q1
    outliers_iqr = ((treino[FEATURES_NUM] < q1 - 1.5 * iqr) | (treino[FEATURES_NUM] > q3 + 1.5 * iqr)).sum()
    display(outliers_iqr.sort_values(ascending=False).to_frame("outliers (IQR)"))



    # 2.4 Matriz de correlação

    corr = treino[FEATURES_NUM + ["quality"]].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True, cbar_kws={"shrink": .8})
    plt.title("Matriz de correlação (Pearson)")
    plt.show()
    display(corr["quality"].drop("quality").sort_values(key=abs, ascending=False).to_frame("corr. com quality"))



    # 2.5 Scatter plots das variáveis mais relacionadas à qualidade

    top4 = corr["quality"].drop("quality").abs().sort_values(ascending=False).index[:4]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, col in zip(axes, top4):
        sns.stripplot(x="quality", y=col, data=treino, ax=ax, alpha=.25, jitter=.3, size=2.5, color="#4a6fa5")
        sns.pointplot(x="quality", y=col, data=treino, ax=ax, color="#c0392b", errorbar=None)
        ax.set_title(f"{col} × quality")
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.scatterplot(x="alcohol", y="density", hue="quality", data=treino, palette="viridis", s=12, ax=ax[0])
    ax[0].set_title("alcohol × density (cor = quality)")
    sns.scatterplot(x="total sulfur dioxide", y="chlorides", hue="type", data=treino, s=12, ax=ax[1],
                    palette={"red": "#8c2d4a", "white": "#d9c27a"})
    ax[1].set_title("Tintos e brancos separam bem em SO2 total × cloretos")
    plt.tight_layout()
    plt.show()



    # 3. Pré-processamento

    class LimitadorQuantis(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
        """Winsorização: corta valores abaixo/acima dos quantis aprendidos no treino."""

        def __init__(self, inferior=0.01, superior=0.99):
            self.inferior = inferior
            self.superior = superior

        def fit(self, X, y=None):
            X = validate_data(self, X, dtype=float)
            self.lim_inf_ = np.nanquantile(X, self.inferior, axis=0)
            self.lim_sup_ = np.nanquantile(X, self.superior, axis=0)
            return self

        def transform(self, X):
            X = validate_data(self, X, reset=False, dtype=float)
            return np.clip(X, self.lim_inf_, self.lim_sup_)


    def preparar_numericas():
        return Pipeline([
            ("imputacao", SimpleImputer(strategy="median")),
            ("outliers", LimitadorQuantis(0.01, 0.99)),
            ("padronizacao", StandardScaler()),
        ])


    def codificar_tipo():
        return OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)


    def criar_preprocessador():
        return ColumnTransformer([
            ("num", preparar_numericas(), FEATURES_NUM),
            ("cat", codificar_tipo(), ["type"]),
        ], verbose_feature_names_out=False)



    # 4. Parte 1 — Regressão (predição da nota)

    # 4.1 Modelos e validação cruzada

    modelos_reg = {
        "Baseline (média)": DummyRegressor(strategy="mean"),
        "Regressão Linear": LinearRegression(),
        "Ridge": Ridge(random_state=SEED),
        "Random Forest": RandomForestRegressor(random_state=SEED, n_jobs=1),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=SEED),
    }
    cv_reg = KFold(n_splits=5, shuffle=True, random_state=SEED)
    metricas_reg = {"RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error", "R2": "r2"}

    linhas = []
    for nome, modelo in modelos_reg.items():
        pipe = Pipeline([("prep", criar_preprocessador()), ("modelo", modelo)])
        with parallel_config(backend="threading"):
            res = cross_validate(pipe, X_train, y_train, cv=cv_reg, scoring=metricas_reg,
                                 n_jobs=N_JOBS, error_score="raise")
        linhas.append({"Modelo": nome,
                       "RMSE": -res["test_RMSE"].mean(), "MAE": -res["test_MAE"].mean(),
                       "R²": res["test_R2"].mean(), "R² (desvio)": res["test_R2"].std()})
    resultado_cv_reg = pd.DataFrame(linhas).set_index("Modelo")
    display(resultado_cv_reg.round(4))



    # 4.2 Ajuste de hiperparâmetros (GridSearchCV)

    grades_reg = {
        "Ridge": (Ridge(random_state=SEED),
                  {"modelo__alpha": [0.01, 0.1, 1, 10, 50, 100]}),
        "Random Forest": (RandomForestRegressor(random_state=SEED, n_jobs=1),
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
    buscas_reg = {}
    for nome, (modelo, grade) in grades_reg.items():
        busca = GridSearchCV(Pipeline([("prep", criar_preprocessador()), ("modelo", modelo)]),
                             grade, cv=cv_reg, scoring="neg_root_mean_squared_error", n_jobs=N_JOBS, error_score="raise")
        with parallel_config(backend="threading"):
            busca.fit(X_train, y_train)
        melhores_reg[nome] = busca.best_estimator_
        buscas_reg[nome] = {"RMSE CV": -busca.best_score_, "parâmetros": busca.best_params_}
        print(f"{nome:22s} RMSE CV = {-busca.best_score_:.4f}  |  {busca.best_params_}")



    # 4.3 Avaliação final no conjunto de teste

    # Congelamos a escolha usando somente as dobras do treino.
    rmse_cv = {nome: busca["RMSE CV"] for nome, busca in buscas_reg.items()}
    rmse_cv["Regressão Linear"] = resultado_cv_reg.loc["Regressão Linear", "RMSE"]
    melhor_nome_reg = min(rmse_cv, key=rmse_cv.get)

    for nome in ("Baseline (média)", "Regressão Linear"):
        pipe = Pipeline([("prep", criar_preprocessador()), ("modelo", modelos_reg[nome])])
        melhores_reg[nome] = pipe.fit(X_train, y_train)
    linhas = []
    for nome, pipe in melhores_reg.items():
        pred = pipe.predict(X_test)
        linhas.append({"Modelo": nome, "RMSE": root_mean_squared_error(y_test, pred),
                       "MAE": mean_absolute_error(y_test, pred), "R²": r2_score(y_test, pred)})
    resultado_teste_reg = pd.DataFrame(linhas).set_index("Modelo")
    display(resultado_teste_reg.round(4))



    print(f"Modelo selecionado no treino: {melhor_nome_reg}")
    melhor_reg = melhores_reg[melhor_nome_reg]
    pred = melhor_reg.predict(X_test)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.boxplot(x=y_test, y=pred, ax=ax[0], color="#a5c8e4")
    notas = np.sort(y_test.unique())
    ax[0].plot(range(len(notas)), notas, "r--", label="previsão perfeita")
    ax[0].set_xlabel("nota real"); ax[0].set_ylabel("nota prevista")
    ax[0].set_title(f"{melhor_nome_reg}: previsto × real"); ax[0].legend()
    sns.histplot(y_test - pred, bins=40, ax=ax[1], color="#4a6fa5")
    ax[1].set_title("Distribuição dos resíduos (real − previsto)")
    plt.tight_layout()
    plt.show()
    residuos = pd.DataFrame({"nota": y_test, "previsto": pred, "erro": y_test - pred})
    resumo_residuos = residuos.groupby("nota")["erro"].agg(["size", "mean"])
    display(resumo_residuos.round(3))



    # 4.4 Interpretação — quais variáveis mais impactam a nota?

    lin = melhores_reg["Regressão Linear"]
    coef = pd.Series(lin.named_steps["modelo"].coef_,
                     index=lin.named_steps["prep"].get_feature_names_out()).sort_values()
    ridge = melhores_reg["Ridge"]
    coef_ridge = pd.Series(ridge.named_steps["modelo"].coef_,
                           index=ridge.named_steps["prep"].get_feature_names_out())

    fig, ax = plt.subplots(figsize=(8, 5))
    coef.plot.barh(ax=ax, color=["#c0392b" if c < 0 else "#27ae60" for c in coef])
    ax.set_title("Coeficientes padronizados — Regressão Linear"); ax.axvline(0, color="k", lw=.8)
    plt.tight_layout()
    plt.show()
    display(pd.DataFrame({"Linear": coef, "Ridge": coef_ridge}).round(3))



    # Importância por permutação no teste: quanto o R² cai ao embaralhar cada variável.
    # Mede a dependência do modelo já ajustado; preditores correlacionados podem compartilhar efeito.
    with parallel_config(backend="threading"):
        imp = permutation_importance(melhor_reg, X_test, y_test, n_repeats=10, random_state=SEED,
                                     scoring="r2", n_jobs=N_JOBS)
    imp_reg = pd.Series(imp.importances_mean, index=X_test.columns).sort_values()
    ax = imp_reg.plot.barh(figsize=(8, 5), color="#4a6fa5",
                      title=f"Importância por permutação — {melhor_nome_reg}")
    plt.xlabel("queda no R²")
    plt.tight_layout()
    plt.show()



    # 4.5 Discussão da regressão

    # 5. Parte 2 — Classificação (faixas de qualidade)

    # 5.1 Criação da variável alvo categórica — testando diferentes faixas

    ESQUEMAS = {
        "A: 0-4 | 5-6 | 7-10": ([0, 4, 6, 10], ["baixa", "média", "alta"]),
        "B: 0-5 | 6 | 7-10":   ([0, 5, 6, 10], ["baixa", "média", "alta"]),
        "C: 0-5 | 6-10":       ([0, 5, 10], ["ruim", "bom"]),
    }


    def discretizar(notas, esquema):
        limites, rotulos = ESQUEMAS[esquema]
        return pd.cut(notas, bins=limites, labels=rotulos, include_lowest=True).astype(str)


    distribuicao_faixas = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, esquema in zip(axes, ESQUEMAS):
        contagens = discretizar(y_train, esquema).value_counts().reindex(ESQUEMAS[esquema][1], fill_value=0)
        distribuicao_faixas[esquema] = contagens.to_dict()
        dist = contagens / len(y_train)
        dist.plot.bar(ax=ax, color="#8c2d4a", rot=0)
        ax.set_title(esquema); ax.set_ylabel("proporção")
        for i, v in enumerate(dist):
            ax.text(i, v + .01, f"{v:.1%}", ha="center")
    plt.tight_layout()
    plt.show()



    # 5.2 Escolha da métrica

    # 5.3 Comparação inicial: 4 classificadores × 3 esquemas

    def classificadores():
        return {
            "Baseline (majoritária)": DummyClassifier(strategy="most_frequent"),
            "Regressão Logística": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "SVM (RBF)": SVC(class_weight="balanced", random_state=SEED),
            "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=1),
            "HistGradientBoosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
        }


    cv_clf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    linhas = []
    for esquema in ESQUEMAS:
        # mesmo split do treino/teste da regressão (mesmos índices)
        y_tr = LabelEncoder().fit(ESQUEMAS[esquema][1]).transform(discretizar(y_train, esquema))
        for nome, modelo in classificadores().items():
            pipe = Pipeline([("prep", criar_preprocessador()), ("modelo", modelo)])
            with parallel_config(backend="threading"):
                res = cross_validate(
                    pipe, X_train, y_tr, cv=cv_clf, n_jobs=N_JOBS, error_score="raise",
                    scoring={"f1": "f1_macro", "bal": "balanced_accuracy", "acc": "accuracy"})
            linhas.append({"Esquema": esquema, "Modelo": nome, "F1-macro": res["test_f1"].mean(),
                           "Acur. balanceada": res["test_bal"].mean(), "Acurácia": res["test_acc"].mean()})
    comparacao = pd.DataFrame(linhas)
    display(comparacao.pivot(index="Modelo", columns="Esquema", values="F1-macro").round(4))



    display(comparacao.set_index(["Esquema", "Modelo"]).round(4))



    # 5.4 Esquema principal e balanceamento de classes

    ESQ = "A: 0-4 | 5-6 | 7-10"
    codificador = LabelEncoder().fit(ESQUEMAS[ESQ][1])
    y_train_c = codificador.transform(discretizar(y_train, ESQ))
    # LabelEncoder numera as classes em ordem alfabética
    print("Classes codificadas:", {str(c): int(i) for c, i in zip(codificador.classes_,
                                                                codificador.transform(codificador.classes_))})
    print("Distribuição no treino:", pd.Series(codificador.inverse_transform(y_train_c)).value_counts().to_dict())

    def criar_pipeline_clf(modelo, amostrador="passthrough"):
        preparo = ColumnTransformer([
            ("num", preparar_numericas(), FEATURES_NUM),
            ("tipo", "passthrough", ["type"]),
        ], verbose_feature_names_out=False).set_output(transform="pandas")
        codificacao = ColumnTransformer(
            [("tipo", codificar_tipo(), ["type"])], remainder="passthrough")
        return ImbPipeline([
            ("prep", preparo),
            ("amostrador", amostrador),
            ("codificacao", codificacao),
            ("modelo", modelo),
        ])


    def criar_smotenc():
        return SMOTENC(categorical_features=["type"], random_state=SEED)


    estrategias = {
        "Sem ajuste": criar_pipeline_clf(RandomForestClassifier(random_state=SEED, n_jobs=1)),
        "Pesos de classe": criar_pipeline_clf(
            RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=1)),
        "SMOTENC": criar_pipeline_clf(
            RandomForestClassifier(random_state=SEED, n_jobs=1), criar_smotenc()),
    }
    linhas = []
    for nome, pipe in estrategias.items():
        with parallel_config(backend="threading"):
            res = cross_validate(pipe, X_train, y_train_c, cv=cv_clf, n_jobs=N_JOBS,
                                 scoring={"f1": "f1_macro", "bal": "balanced_accuracy"},
                                 error_score="raise")
        linhas.append({"Estratégia": nome, "F1-macro": res["test_f1"].mean(),
                       "Acur. balanceada": res["test_bal"].mean()})
    resultado_balanceamento = pd.DataFrame(linhas).set_index("Estratégia")
    display(resultado_balanceamento.round(4))



    # 5.5 Ajuste de hiperparâmetros (GridSearchCV, F1-macro)

    grades_clf = {
        "Regressão Logística": (
            LogisticRegression(max_iter=3000),
            {"amostrador": ["passthrough", criar_smotenc()],
             "modelo__C": [0.01, 0.1, 1, 10],
             "modelo__class_weight": [None, "balanced"]}),
        "SVM (RBF)": (
            SVC(class_weight="balanced", random_state=SEED),
            {"modelo__C": [1, 10, 30], "modelo__gamma": ["scale", 0.1, 0.3]}),
        "Random Forest": (
            RandomForestClassifier(random_state=SEED, n_jobs=1),
            {"amostrador": ["passthrough", criar_smotenc()],
             "modelo__class_weight": [None, "balanced_subsample"],
             "modelo__n_estimators": [200],
             "modelo__max_depth": [None, 15],
             "modelo__min_samples_leaf": [1, 3]}),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
            {"modelo__learning_rate": [0.05, 0.1],
             "modelo__max_iter": [200, 400],
             "modelo__max_leaf_nodes": [15, 31]}),
    }

    melhores_clf = {}
    buscas_clf = {}
    for nome, (modelo, grade) in grades_clf.items():
        busca = GridSearchCV(criar_pipeline_clf(modelo), grade, cv=cv_clf, scoring="f1_macro",
                             n_jobs=N_JOBS, error_score="raise")
        with parallel_config(backend="threading"):
            busca.fit(X_train, y_train_c)
        melhores_clf[nome] = busca.best_estimator_
        parametros = {k: "SMOTENC" if isinstance(v, SMOTENC) else v
                      for k, v in busca.best_params_.items()}
        buscas_clf[nome] = {"F1-macro CV": busca.best_score_, "parâmetros": parametros}
        print(f"{nome:22s} F1-macro CV = {busca.best_score_:.4f}  |  {parametros}")

    melhor_nome_clf = max(buscas_clf, key=lambda nome: buscas_clf[nome]["F1-macro CV"])
    melhores_clf["Baseline (majoritária)"] = criar_pipeline_clf(
        DummyClassifier(strategy="most_frequent")).fit(X_train, y_train_c)



    # 5.6 Avaliação final no conjunto de teste

    y_test_c = codificador.transform(discretizar(y_test, ESQ))
    linhas = []
    for nome, pipe in melhores_clf.items():
        pred_c = pipe.predict(X_test)
        linhas.append({"Modelo": nome, "F1-macro": f1_score(y_test_c, pred_c, average="macro"),
                       "Acur. balanceada": balanced_accuracy_score(y_test_c, pred_c),
                       "Acurácia": (pred_c == y_test_c).mean()})
    resultado_teste_clf = pd.DataFrame(linhas).set_index("Modelo")
    display(resultado_teste_clf.round(4))



    melhor_clf = melhores_clf[melhor_nome_clf]
    pred_c = melhor_clf.predict(X_test)
    print(f"Modelo selecionado no treino: {melhor_nome_clf}\n")
    ordem = ESQUEMAS[ESQ][1]
    idx = codificador.transform(ordem)
    print(classification_report(y_test_c, pred_c, labels=idx, target_names=ordem,
                                digits=3, zero_division=0))
    relatorio_classes = classification_report(y_test_c, pred_c, labels=idx, target_names=ordem,
                                              output_dict=True, zero_division=0)
    matriz_confusao = confusion_matrix(y_test_c, pred_c, labels=idx)
    assert matriz_confusao.sum() == len(y_test)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ConfusionMatrixDisplay.from_predictions(y_test_c, pred_c, labels=idx, display_labels=ordem,
                                            cmap="Blues", ax=ax[0], colorbar=False)
    ax[0].set_title("Matriz de confusão (contagens)")
    ConfusionMatrixDisplay.from_predictions(y_test_c, pred_c, labels=idx, display_labels=ordem,
                                            normalize="true", values_format=".0%", cmap="Blues",
                                            ax=ax[1], colorbar=False)
    ax[1].set_title("Normalizada por linha (recall de cada classe)")
    plt.tight_layout()
    plt.show()



    # Matrizes de confusão de todos os modelos ajustados, para comparar os padrões de erro
    fig, axes = plt.subplots(1, len(melhores_clf), figsize=(5 * len(melhores_clf), 4.5))
    for ax, (nome, pipe) in zip(axes, melhores_clf.items()):
        ConfusionMatrixDisplay.from_predictions(y_test_c, pipe.predict(X_test), labels=idx,
                                                display_labels=ordem, normalize="true",
                                                values_format=".0%", cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(nome)
    plt.tight_layout()
    plt.show()



    # Quais notas originais são mais confundidas? (erros do melhor modelo por nota real)
    erros = pd.DataFrame({"nota": y_test.values, "real": codificador.inverse_transform(y_test_c),
                          "previsto": codificador.inverse_transform(pred_c)})
    erros["acertou"] = erros["real"] == erros["previsto"]
    erros_por_nota = erros.groupby("nota")["acertou"].agg(n="count", acerto="mean")
    display(erros_por_nota.round(3))



    with parallel_config(backend="threading"):
        imp = permutation_importance(melhor_clf, X_test, y_test_c, n_repeats=10, random_state=SEED,
                                     scoring="f1_macro", n_jobs=N_JOBS)
    imp_clf = pd.Series(imp.importances_mean, index=X_test.columns).sort_values()
    ax = imp_clf.plot.barh(figsize=(8, 5), color="#8c2d4a",
                      title=f"Importância por permutação — {melhor_nome_clf}")
    plt.xlabel("queda no F1-macro")
    plt.tight_layout()
    plt.show()



    # 5.7 Discussão da classificação

    # 6. Conclusões


if __name__ == "__main__":
    main()
