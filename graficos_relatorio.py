"""Figuras da execução atual e imagens já presentes no notebook original."""
import base64
import io
import json
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
CACHE = RAIZ / "resultados" / ".matplotlib"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE))

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, root_mean_squared_error, f1_score
from sklearn.model_selection import train_test_split

from limpeza import COLUNAS_NUMERICAS


def figura(fig, legenda):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    imagem = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f'<figure><img src="data:image/png;base64,{imagem}" alt="{legenda}"><figcaption>{legenda}</figcaption></figure>'


def graficos_atuais(metricas):
    """Reconstrói o teste e confere métricas antes de atribuir gráficos aos modelos."""
    dados = pd.read_csv(RAIZ / 'dados/processados/vinhos_limpos.csv')
    X = dados[COLUNAS_NUMERICAS + ['type']]
    y = dados['quality']
    treino, teste, ytreino, yteste = train_test_split(X, y, test_size=.2, stratify=y, random_state=metricas['semente'])
    reg = joblib.load(RAIZ / 'modelos/regressao.joblib')
    clf = joblib.load(RAIZ / 'modelos/classificacao.joblib')
    previstas = reg.predict(teste)
    reais = pd.cut(yteste, bins=[-1,4,6,10], labels=['baixa','media','alta']).astype(str)
    categorias = clf.predict(teste)
    if not np.isclose(root_mean_squared_error(yteste, previstas), metricas['tarefas']['regressao']['metricas_teste']['rmse'], atol=1e-8) or not np.isclose(f1_score(reais, categorias, average='macro'), metricas['tarefas']['classificacao']['metricas_teste']['f1_macro'], atol=1e-8):
        raise ValueError('Os modelos carregados não correspondem às métricas salvas. Execute novamente a etapa 2.')
    imagens = []
    fig, ax = plt.subplots(figsize=(7, 3.8))
    contagem = ytreino.value_counts().sort_index()
    ax.bar(contagem.index.astype(str), contagem.values, color='#83405a')
    ax.set(xlabel='Nota de qualidade', ylabel='Quantidade de vinhos', title='Distribuição da nota no treino')
    imagens.append(figura(fig, 'Figura 1 — Distribuição das notas no conjunto de treino.'))
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = dados.loc[treino.index, COLUNAS_NUMERICAS + ['quality']].corr()
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(len(corr)), corr.columns, rotation=50, ha='right', fontsize=8)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=8)
    ax.set_title('Correlação de Pearson no treino')
    fig.colorbar(im, ax=ax, label='Correlação')
    imagens.append(figura(fig, 'Figura 2 — Matriz de correlação dos atributos e da nota no treino.'))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    rng=np.random.default_rng(42)
    ax.scatter(yteste + rng.uniform(-.12,.12,len(yteste)), previstas, alpha=.25, s=12, color='#38678c')
    ax.plot([yteste.min(), yteste.max()], [yteste.min(), yteste.max()], '--', color='#b33b36', label='Previsão perfeita')
    ax.set(xlabel='Nota real', ylabel='Nota prevista', title='Regressão: previsto × real no teste')
    ax.legend()
    imagens.append(figura(fig, 'Figura 3 — Previsões do modelo de regressão no conjunto de teste.'))
    fig, ax = plt.subplots(figsize=(6, 4.8))
    ordem=['baixa','media','alta']
    matriz=confusion_matrix(reais, categorias, labels=ordem)
    im=ax.imshow(matriz, cmap='Blues')
    ax.set_xticks(range(3), ['baixa','média','alta']); ax.set_yticks(range(3), ['baixa','média','alta'])
    ax.set(xlabel='Classe prevista', ylabel='Classe real', title='Matriz de confusão no teste')
    for i in range(3):
        for j in range(3): ax.text(j,i,str(matriz[i,j]),ha='center',va='center',color='white' if matriz[i,j]>matriz.max()/2 else 'black')
    fig.colorbar(im, ax=ax, label='Quantidade')
    imagens.append(figura(fig, 'Figura 4 — Matriz de confusão do modelo de classificação no teste.'))
    return imagens


def graficos_notebook():
    """Recupera as imagens já executadas; não altera nem executa o notebook."""
    arquivo = RAIZ / 'trabalho_ml_vinhos.ipynb'
    if not arquivo.exists():
        return ''
    nb = json.loads(arquivo.read_text(encoding='utf-8'))
    nomes = ['Distribuição das notas e dos tipos', 'Histogramas dos atributos',
             'Boxplots dos atributos', 'Matriz de correlação', 'Atributos relacionados à qualidade',
             'Relações entre atributos e tipos de vinho', 'Regressão: previsões e resíduos',
             'Coeficientes da regressão linear', 'Importância por permutação na regressão',
             'Distribuição dos esquemas de classificação', 'Matriz de confusão do classificador selecionado',
             'Matrizes dos classificadores comparados', 'Importância por permutação na classificação']
    imagens = [o['data']['image/png'] for c in nb['cells'] for o in c.get('outputs', []) if 'image/png' in o.get('data', {})]
    figuras=[]
    for i, imagem in enumerate(imagens):
        imagem = ''.join(imagem) if isinstance(imagem, list) else imagem
        nome = nomes[i] if i < len(nomes) else 'Gráfico adicional do notebook'
        figuras.append(f'<figure><img src="data:image/png;base64,{imagem}" alt="{nome}"><figcaption>Figura histórica {i+1} — {nome}.</figcaption></figure>')
    return ''.join(figuras)
