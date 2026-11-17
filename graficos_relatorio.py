"""Gráficos do relatório, todos em português, com legenda e texto de apoio.

Os gráficos são reconstruídos a partir da base limpa e dos modelos salvos; nada é
treinado de novo. Antes de desenhar, confere se os modelos carregados reproduzem as
métricas de resultados/metricas.json.
"""
import base64
import html
import io
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
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, f1_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from limpeza import COLUNAS_NUMERICAS

# Nomes em português (com unidade) para cada variável da base.
NOMES = {
    'fixed acidity': 'Acidez fixa (g/L)',
    'volatile acidity': 'Acidez volátil (g/L)',
    'citric acid': 'Ácido cítrico (g/L)',
    'residual sugar': 'Açúcar residual (g/L)',
    'chlorides': 'Cloretos (g/L)',
    'free sulfur dioxide': 'Dióxido de enxofre livre (mg/L)',
    'total sulfur dioxide': 'Dióxido de enxofre total (mg/L)',
    'density': 'Densidade (g/cm³)',
    'pH': 'pH',
    'sulphates': 'Sulfatos (g/L)',
    'alcohol': 'Teor alcoólico (% vol.)',
    'type': 'Tipo do vinho',
    'quality': 'Nota de qualidade',
}
CURTOS = {
    'fixed acidity': 'Acidez fixa', 'volatile acidity': 'Acidez volátil', 'citric acid': 'Ácido cítrico',
    'residual sugar': 'Açúcar residual', 'chlorides': 'Cloretos', 'free sulfur dioxide': 'SO₂ livre',
    'total sulfur dioxide': 'SO₂ total', 'density': 'Densidade', 'pH': 'pH', 'sulphates': 'Sulfatos',
    'alcohol': 'Teor alcoólico', 'type': 'Tipo do vinho', 'quality': 'Nota',
}

# Cores consistentes em todo o relatório (distinguíveis também em escala de cinza).
COR_BAIXA, COR_MEDIA, COR_ALTA = '#c0504d', '#8c8c8c', '#2e7d4f'
COR_TINTO, COR_BRANCO = '#8e2c48', '#d4a72c'
COR_PRINCIPAL = '#38678c'
FAIXAS = [('baixa', 'Baixa (notas 0 a 4)', COR_BAIXA), ('media', 'Média (notas 5 e 6)', COR_MEDIA),
          ('alta', 'Alta (notas 7 a 10)', COR_ALTA)]
ROTULO_FAIXA = {'baixa': 'Baixa', 'media': 'Média', 'alta': 'Alta'}

plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': .3, 'axes.axisbelow': True})


def figura(fig, titulo, explicacao):
    """Converte a figura em HTML com título numerado pelo chamador e texto 'Como ler'."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    imagem = base64.b64encode(buffer.getvalue()).decode('ascii')
    t, e = html.escape(titulo), html.escape(explicacao)
    return (f'<figure><img src="data:image/png;base64,{imagem}" alt="{t}">'
            f'<figcaption><strong>{t}.</strong> <em>Como ler:</em> {e}</figcaption></figure>')


def _faixa(notas):
    return pd.cut(notas, bins=[-1, 4, 6, 10], labels=['baixa', 'media', 'alta']).astype(str)


def _legenda_faixas(ax, **kw):
    ax.legend(handles=[Patch(color=c, label=r) for _, r, c in FAIXAS], title='Faixa de qualidade', **kw)


def _distribuicao(ytreino, tipos):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={'width_ratios': [2.2, 1]})
    contagem = ytreino.value_counts().sort_index()
    cores = [COR_BAIXA if n <= 4 else COR_MEDIA if n <= 6 else COR_ALTA for n in contagem.index]
    barras = a.bar(contagem.index.astype(str), contagem.values, color=cores)
    a.bar_label(barras, padding=2, fontsize=9)
    a.set(xlabel='Nota de qualidade (0 a 10)', ylabel='Quantidade de vinhos',
          title='Quantos vinhos há em cada nota')
    a.set_ylim(0, contagem.max() * 1.12)
    _legenda_faixas(a, loc='upper left')
    t = tipos.map({'red': 'Tinto', 'white': 'Branco'}).value_counts()
    barras = b.bar(t.index, t.values, color=[COR_TINTO if i == 'Tinto' else COR_BRANCO for i in t.index])
    b.bar_label(barras, padding=2, fontsize=9)
    b.set(xlabel='Tipo do vinho', ylabel='Quantidade de vinhos', title='Tintos e brancos')
    b.set_ylim(0, t.max() * 1.12)
    fig.tight_layout()
    return figura(fig, 'Distribuição das notas e dos tipos de vinho (conjunto de treino)',
                  'cada barra é a quantidade de vinhos com aquela nota; as cores indicam a faixa usada na classificação. '
                  'Quase todos os vinhos têm nota 5 ou 6, e por isso as faixas baixa e alta são raras e mais difíceis de prever.')


def _histogramas(treino):
    fig, eixos = plt.subplots(3, 4, figsize=(13, 8.5))
    eixos = eixos.ravel()
    for ax, col in zip(eixos, COLUNAS_NUMERICAS):
        for tipo, rotulo, cor in (('red', 'Tinto', COR_TINTO), ('white', 'Branco', COR_BRANCO)):
            ax.hist(treino.loc[treino['type'] == tipo, col].dropna(), bins=30, density=True, alpha=.6, color=cor, label=rotulo)
        ax.set_title(CURTOS[col], fontsize=10)
        ax.set_yticks([])
        ax.set_xlabel(NOMES[col].split('(')[-1].rstrip(')') if '(' in NOMES[col] else 'valor', fontsize=8)
    eixos[-1].axis('off')
    h, l = eixos[0].get_legend_handles_labels()
    eixos[-1].legend(h, l, title='Tipo do vinho', loc='center', fontsize=11)
    fig.suptitle('Distribuição de cada atributo físico-químico, por tipo de vinho', fontweight='bold', y=1.0)
    fig.tight_layout()
    return figura(fig, 'Histogramas dos atributos físico-químicos (conjunto de treino)',
                  'cada gráfico mostra como os valores de um atributo se espalham; a altura indica a concentração de vinhos (as áreas '
                  'dos dois tipos são comparáveis). Onde as cores se separam, o atributo ajuda a distinguir tintos de brancos.')


def _correlacao_com_nota(corr):
    c = corr['quality'].drop('quality').sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    barras = ax.barh([CURTOS[i] for i in c.index], c.values, color=[COR_ALTA if v > 0 else COR_BAIXA for v in c.values])
    ax.bar_label(barras, labels=[f'{v:+.2f}'.replace('.', ',') for v in c.values], padding=3, fontsize=9)
    ax.axvline(0, color='black', lw=.8)
    ax.set(xlabel='Correlação de Pearson com a nota (de −1 a +1)', title='Quais atributos se associam à nota')
    ax.set_xlim(min(c.min() * 1.3, -.1), max(c.max() * 1.3, .1))
    ax.legend(handles=[Patch(color=COR_ALTA, label='Quanto maior o atributo, maior tende a ser a nota'),
                       Patch(color=COR_BAIXA, label='Quanto maior o atributo, menor tende a ser a nota')],
              loc='lower right', fontsize=8.5)
    fig.tight_layout()
    return figura(fig, 'Correlação de cada atributo com a nota (conjunto de treino)',
                  'barras para a direita indicam que a nota tende a subir junto com o atributo; para a esquerda, que tende a cair. '
                  'Quanto mais longa a barra, mais forte a associação. Correlação não prova causa e efeito.')


def _matriz_correlacao(corr):
    fig, ax = plt.subplots(figsize=(10, 8.2))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
    nomes = [CURTOS[c] for c in corr.columns]
    ax.set_xticks(range(len(nomes)), nomes, rotation=45, ha='right')
    ax.set_yticks(range(len(nomes)), nomes)
    ax.grid(False)
    for i in range(len(nomes)):
        for j in range(len(nomes)):
            v = corr.iloc[i, j]
            ax.text(j, i, f'{v:.2f}'.replace('.', ','), ha='center', va='center', fontsize=8,
                    color='white' if abs(v) > .55 else 'black')
    cb = fig.colorbar(im, ax=ax, shrink=.8)
    cb.set_label('Correlação de Pearson (vermelho: sobem juntos; azul: um sobe e o outro desce)')
    ax.set_title('Correlação entre os atributos e a nota')
    fig.tight_layout()
    return figura(fig, 'Matriz de correlação entre os atributos (conjunto de treino)',
                  'cada célula mostra a correlação entre dois atributos (a diagonal é sempre 1). Vermelho forte: os dois crescem juntos; '
                  'azul forte: quando um cresce, o outro cai. Ex.: densidade e teor alcoólico têm relação inversa.')


def _comparacao_modelos(metricas):
    reg, clf = metricas['tarefas']['regressao'], metricas['tarefas']['classificacao']
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, tarefa, titulo, eixo, menor in ((a, reg, 'Regressão (prever a nota)', 'Erro médio RMSE (menor é melhor)', True),
                                             (b, clf, 'Classificação (prever a faixa)', 'F1 macro (maior é melhor)', False)):
        nomes = list(tarefa['validacao_cruzada'])
        valores = [tarefa['validacao_cruzada'][n] for n in nomes]
        melhor = tarefa['modelo']
        cores = [COR_PRINCIPAL if n == melhor else '#b0b0b0' for n in nomes]
        barras = ax.bar([n.replace('Regressao logistica', 'Regressão logística') for n in nomes], valores, color=cores)
        ax.bar_label(barras, labels=[f'{v:.3f}'.replace('.', ',') for v in valores], padding=2)
        ax.set(ylabel=eixo, title=titulo)
        ax.set_ylim(0, max(valores) * 1.2)
        ax.legend(handles=[Patch(color=COR_PRINCIPAL, label='Modelo escolhido'), Patch(color='#b0b0b0', label='Modelo descartado')],
                  loc='lower right' if menor else 'upper left', fontsize=8.5)
    fig.tight_layout()
    return figura(fig, 'Comparação dos modelos na validação cruzada (apenas dados de treino)',
                  'o treino é dividido em 5 partes e cada modelo é testado em rodízio. Na regressão vence o menor erro; na classificação, '
                  'o maior F1 macro. A barra azul é o modelo escolhido e salvo.')


def _previsto_real(yteste, previstas):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4))
    notas = sorted(yteste.unique())
    dados = [previstas[(yteste == n).to_numpy()] for n in notas]
    a.boxplot(dados, tick_labels=[str(n) for n in notas], patch_artist=True, boxprops=dict(facecolor='#a9c4dc'),
              medianprops=dict(color='black'), flierprops=dict(marker='o', markersize=3, alpha=.5))
    a.plot(range(1, len(notas) + 1), notas, '--', color=COR_BAIXA, label='Previsão perfeita (previsto = real)')
    a.plot([], [], color='black', label='Mediana das previsões')
    a.set(xlabel='Nota real do vinho', ylabel='Nota prevista pelo modelo', title='Nota prevista para cada nota real')
    a.legend(loc='upper left', fontsize=8.5)
    residuo = yteste.to_numpy() - previstas
    b.hist(residuo, bins=40, color=COR_PRINCIPAL, alpha=.8)
    b.axvline(0, color=COR_BAIXA, ls='--', label='Erro zero (acerto exato)')
    b.axvline(residuo.mean(), color='black', label=f'Média dos erros ({residuo.mean() + 0.0:.2f})'.replace('-0.00', '0.00').replace('.', ','))
    b.set(xlabel='Erro = nota real − nota prevista (em pontos)', ylabel='Quantidade de vinhos',
          title='Distribuição dos erros da regressão')
    b.legend(fontsize=8.5)
    fig.tight_layout()
    return figura(fig, 'Regressão: notas previstas e erros no conjunto de teste',
                  'à esquerda, cada caixa reúne as previsões dos vinhos de uma mesma nota real; quanto mais perto da linha tracejada, '
                  'melhor. O modelo aproxima as previsões da média (notas extremas são mal previstas). À direita, erros positivos '
                  'indicam que o modelo previu nota menor que a real.')


def _importancia(modelo, teste, alvo, scoring, titulo, eixo, explicacao, cor, titulo_fig):
    res = permutation_importance(modelo, teste, alvo, scoring=scoring, n_repeats=10, random_state=42, n_jobs=1)
    ordem = np.argsort(res.importances_mean)
    nomes = [CURTOS[teste.columns[i]] for i in ordem]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(nomes, res.importances_mean[ordem], xerr=res.importances_std[ordem], color=cor,
            error_kw=dict(ecolor='black', capsize=3, lw=1), label='Queda média (barra preta: variação entre 10 repetições)')
    ax.set(xlabel=eixo, title=titulo)
    ax.legend(loc='lower right', fontsize=8.5)
    fig.tight_layout()
    return figura(fig, titulo_fig, explicacao)


def _matriz_confusao(reais, categorias):
    ordem = ['baixa', 'media', 'alta']
    rot = [ROTULO_FAIXA[o] for o in ordem]
    m = confusion_matrix(reais, categorias, labels=ordem)
    pct = m / m.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    im = ax.imshow(pct, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(3), rot)
    ax.set_yticks(range(3), rot)
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{m[i, j]}\n({100 * pct[i, j]:.0f}%)'.replace('.', ','), ha='center', va='center',
                    color='white' if pct[i, j] > .5 else 'black', fontsize=11)
    ax.set(xlabel='Faixa prevista pelo modelo', ylabel='Faixa real do vinho', title='Matriz de confusão no teste')
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('Proporção de cada faixa real')
    fig.text(.5, -.02, 'Faixas: baixa = notas 0 a 4; média = 5 e 6; alta = 7 a 10. Cada célula: quantidade (e % da linha).',
             ha='center', fontsize=8.5)
    fig.tight_layout()
    return figura(fig, 'Matriz de confusão da classificação (conjunto de teste)',
                  'cada linha é a faixa real e cada coluna a faixa prevista. A diagonal (de cima à esquerda para baixo à direita) são os acertos. '
                  'A porcentagem é calculada sobre o total da linha, ou seja, mostra que fração de cada faixa real foi encontrada.')


def _metricas_classe(cm):
    ordem = ['baixa', 'media', 'alta']
    fig, ax = plt.subplots(figsize=(9, 4.4))
    largura = .26
    x = np.arange(3)
    grupos = (('precision', 'Precisão (dos que o modelo marcou, quantos acertou)', '#6a8caf'),
              ('recall', 'Recall (dos que existiam, quantos o modelo achou)', '#e0a458'),
              ('f1-score', 'F1 (equilíbrio entre os dois)', '#5a9e6f'))
    for k, (chave, rotulo, cor) in enumerate(grupos):
        valores = [cm['por_classe'][o][chave] for o in ordem]
        barras = ax.bar(x + (k - 1) * largura, valores, largura, color=cor, label=rotulo)
        ax.bar_label(barras, labels=[f'{100 * v:.0f}%' for v in valores], padding=2, fontsize=8.5)
    ax.set_xticks(x, [f'{ROTULO_FAIXA[o]}\n({int(cm["por_classe"][o]["support"])} vinhos no teste)' for o in ordem])
    ax.set(ylabel='Valor (0% a 100%)', ylim=(0, 1.2), title='Desempenho por faixa de qualidade no teste')
    ax.set_yticks(np.arange(0, 1.01, .25), ['0%', '25%', '50%', '75%', '100%'])
    ax.legend(loc='upper left', fontsize=8.5)
    fig.tight_layout()
    return figura(fig, 'Precisão, recall e F1 por faixa (conjunto de teste)',
                  'quanto mais alta a barra, melhor. A faixa média tem muitos exemplos e vai bem; as faixas baixa e alta têm poucos e o '
                  'modelo deixa escapar boa parte delas (recall baixo). Por isso a acurácia geral engana e usamos o F1 macro.')


def graficos_atuais(metricas):
    """Devolve {nome: html} com todas as figuras. Confere as métricas antes de desenhar."""
    dados = pd.read_csv(RAIZ / 'dados/processados/vinhos_limpos.csv')
    X = dados[COLUNAS_NUMERICAS + ['type']]
    y = dados['quality']
    treino, teste, ytreino, yteste = train_test_split(X, y, test_size=.2, stratify=y, random_state=metricas['semente'])
    reg = joblib.load(RAIZ / 'modelos/regressao.joblib')
    clf = joblib.load(RAIZ / 'modelos/classificacao.joblib')
    previstas = reg.predict(teste)
    reais = _faixa(yteste)
    categorias = clf.predict(teste)
    if (not np.isclose(root_mean_squared_error(yteste, previstas), metricas['tarefas']['regressao']['metricas_teste']['rmse'], atol=1e-8)
            or not np.isclose(f1_score(reais, categorias, average='macro'), metricas['tarefas']['classificacao']['metricas_teste']['f1_macro'], atol=1e-8)):
        raise ValueError('Os modelos carregados não correspondem às métricas salvas. Execute novamente a etapa 2.')

    base_treino = dados.loc[treino.index]
    corr = base_treino[COLUNAS_NUMERICAS + ['quality']].corr()
    cm = metricas['tarefas']['classificacao']['metricas_teste']
    figuras = {
        'distribuicao': _distribuicao(ytreino, treino['type']),
        'histogramas': _histogramas(base_treino),
        'corr_nota': _correlacao_com_nota(corr),
        'corr_matriz': _matriz_correlacao(corr),
        'comparacao': _comparacao_modelos(metricas),
        'previsto_real': _previsto_real(yteste, previstas),
        'imp_reg': _importancia(
            reg, teste, yteste, 'r2', 'Importância das variáveis na regressão',
            'Queda do R² ao embaralhar a variável',
            'cada barra mede o quanto o desempenho do modelo piora quando os valores daquela variável são embaralhados no teste. '
            'Barras maiores indicam variáveis das quais o modelo mais depende; variáveis correlacionadas podem dividir a importância.',
            COR_PRINCIPAL, 'Importância das variáveis na regressão (conjunto de teste)'),
        'matriz_confusao': _matriz_confusao(reais, categorias),
        'metricas_classe': _metricas_classe(cm),
        'imp_clf': _importancia(
            clf, teste, reais, 'f1_macro', 'Importância das variáveis na classificação',
            'Queda do F1 macro ao embaralhar a variável',
            'mesma ideia do gráfico da regressão, mas medindo a queda do F1 macro. Quanto maior a barra, mais o modelo precisa '
            'daquela variável para separar as faixas de qualidade.',
            COR_ALTA, 'Importância das variáveis na classificação (conjunto de teste)'),
    }
    return figuras
