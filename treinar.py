"""Treina e avalia os modelos usando o CSV já limpo."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             f1_score, mean_absolute_error, r2_score, root_mean_squared_error)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from limpeza import COLUNAS_NUMERICAS, ler_csv

RAIZ = Path(__file__).resolve().parent
LIMPO = RAIZ / 'dados/processados/vinhos_limpos.csv'
RESULTADOS = RAIZ / 'resultados'
MODELOS = RAIZ / 'modelos'
SEMENTE = 42


def pipeline(estimator):
    numerico = Pipeline([('imputacao', SimpleImputer(strategy='median')),
                         ('escala', StandardScaler())])
    preparo = ColumnTransformer([
        ('numerico', numerico, COLUNAS_NUMERICAS),
        ('tipo', OneHotEncoder(handle_unknown='ignore'), ['type']),
    ])
    return Pipeline([('preparo', preparo), ('modelo', estimator)])


def main():
    if not LIMPO.exists():
        raise FileNotFoundError(f"Base limpa não encontrada: {LIMPO}. Execute primeiro limpar_base.py.")
    dados = ler_csv(LIMPO)
    if not dados['type'].isin(['red', 'white']).all():
        raise ValueError("A base limpa deve conter somente tipos red e white.")
    quality = pd.to_numeric(dados['quality'], errors='coerce')
    if not (quality.between(0, 10) & quality.mod(1).eq(0)).all():
        raise ValueError("A base limpa deve conter notas inteiras de 0 a 10.")
    dados['quality'] = quality.astype(int)
    for pasta in (RESULTADOS, MODELOS):
        pasta.mkdir(parents=True, exist_ok=True)
    print(f'Base limpa utilizada: {LIMPO} ({len(dados)} linhas)', flush=True)

    X = dados[COLUNAS_NUMERICAS + ['type']]
    y = dados['quality']
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEMENTE)
    classes_treino = pd.cut(y_treino, bins=[-1, 4, 6, 10], labels=['baixa', 'media', 'alta']).astype(str)
    classes_teste = pd.cut(y_teste, bins=[-1, 4, 6, 10], labels=['baixa', 'media', 'alta']).astype(str)
    tarefas = {
        'regressao': (y_treino, y_teste, {
            'Ridge': Ridge(),
            'Random Forest': RandomForestRegressor(n_estimators=150, min_samples_leaf=2, random_state=SEMENTE, n_jobs=1),
        }, KFold(n_splits=5, shuffle=True, random_state=SEMENTE), 'neg_root_mean_squared_error'),
        'classificacao': (classes_treino, classes_teste, {
            'Regressao logistica': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEMENTE),
            'Random Forest': RandomForestClassifier(n_estimators=150, min_samples_leaf=2, class_weight='balanced', random_state=SEMENTE, n_jobs=1),
        }, StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE), 'f1_macro'),
    }
    resumo = {'semente': SEMENTE, 'linhas_treino': len(X_treino), 'linhas_teste': len(X_teste),
              'sha256_csv_limpo': hashlib.sha256(LIMPO.read_bytes()).hexdigest(),
              'versoes': {'numpy': np.__version__, 'pandas': pd.__version__, 'scikit-learn': sklearn.__version__, 'joblib': joblib.__version__},
              'tarefas': {}}
    for tarefa, (y_tr, y_te, candidatos, cv, metrica) in tarefas.items():
        pontuacoes = {}
        for nome, estimador in candidatos.items():
            print(f'{tarefa}: avaliando {nome}...', flush=True)
            scores = cross_val_score(pipeline(estimador), X_treino, y_tr, cv=cv, scoring=metrica, n_jobs=1, error_score='raise')
            pontuacoes[nome] = float(-scores.mean() if tarefa == 'regressao' else scores.mean())
        escolhido = (min if tarefa == 'regressao' else max)(pontuacoes, key=pontuacoes.get)
        modelo = pipeline(candidatos[escolhido]).fit(X_treino, y_tr)
        previsao = modelo.predict(X_teste)
        if tarefa == 'regressao':
            metricas = {'rmse': float(root_mean_squared_error(y_te, previsao)),
                        'mae': float(mean_absolute_error(y_te, previsao)), 'r2': float(r2_score(y_te, previsao))}
        else:
            metricas = {'f1_macro': float(f1_score(y_te, previsao, average='macro')),
                        'acuracia_balanceada': float(balanced_accuracy_score(y_te, previsao)),
                        'acuracia': float(accuracy_score(y_te, previsao)),
                        'por_classe': classification_report(y_te, previsao, output_dict=True, zero_division=0)}
        joblib.dump(modelo, MODELOS / f'{tarefa}.joblib')
        resumo['tarefas'][tarefa] = {'modelo': escolhido, 'metrica_cv': 'RMSE (menor e melhor)' if tarefa == 'regressao' else 'F1 macro (maior e melhor)',
                                     'validacao_cruzada': pontuacoes, 'metricas_teste': metricas}
        print(f'{tarefa}: selecionado {escolhido}', flush=True)
    (RESULTADOS / 'metricas.json').write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Modelos e métricas salvos.', flush=True)


if __name__ == '__main__':
    main()
