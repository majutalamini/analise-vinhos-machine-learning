"""Gera previsões a partir de um CSV já tratado com as mesmas regras do treino."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from limpeza import COLUNAS_NUMERICAS

RAIZ = Path(__file__).resolve().parent
COLUNAS = COLUNAS_NUMERICAS + ['type']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('entrada', type=Path, help='CSV tratado com as 12 variáveis de entrada')
    parser.add_argument('saida', type=Path, help='caminho para salvar previsões CSV')
    args = parser.parse_args()
    if args.entrada.resolve() == args.saida.resolve():
        parser.error('A saída deve ser diferente da entrada.')
    dados = pd.read_csv(args.entrada)
    faltantes = sorted(set(COLUNAS) - set(dados.columns))
    if faltantes:
        parser.error(f'Colunas ausentes: {faltantes}')
    for coluna in COLUNAS_NUMERICAS:
        dados[coluna] = pd.to_numeric(dados[coluna], errors='raise')
    if not dados['type'].isin(['red', 'white']).all():
        parser.error("A coluna type deve conter apenas 'red' ou 'white'.")
    X = dados[COLUNAS]
    for tarefa in ('regressao', 'classificacao'):
        modelo = joblib.load(RAIZ / 'modelos' / f'{tarefa}.joblib')
        dados[f'previsao_{tarefa}'] = modelo.predict(X)
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    dados.to_csv(args.saida, index=False)
    print(f'Previsões salvas em {args.saida}')


if __name__ == '__main__':
    main()
