"""
limpeza.py — Limpeza do Wine Quality Dataset entregue pelo professor.

O arquivo de cada equipe vem com "sujeiras" propositais/típicas de exportação
via Excel. Este módulo corrige o que é recuperável e marca como NaN o que não é,
deixando a imputação para o pipeline (para não vazar informação do teste).

Problemas tratados:
1. `alcohol` lido como texto: valores como '100.333.333.333.333' (separador de
   milhar inserido pelo Excel) ou '9.39.39.39...' (texto repetido).
2. Casas decimais perdidas: ex. volatile acidity 675 -> 0.675,
   chlorides 76 -> 0.076, density 9.978 / 998 -> 0.9978 / 0.998.
3. `type` com texto repetido ('whitewhite...') ou ausente.
4. `quality` ausente ou fora da escala 0-10 (ex.: 238) -> linha descartada,
   pois o alvo não pode ser imputado.
5. Valores impossíveis de reescalar para a faixa plausível -> NaN.

Uso:
    from limpeza import carregar_e_limpar
    df, relatorio = carregar_e_limpar("group_X_winequality.csv")
ou pela linha de comando:
    python limpeza.py entrada.csv saida_limpa.csv
"""
import re
import sys

import numpy as np
import pandas as pd

# Faixas plausíveis por variável (baseadas no dataset original da UCI,
# com uma pequena folga). Usadas para decidir se um valor perdeu a vírgula.
FAIXAS = {
    "fixed acidity":        (3.5, 16.5),
    "volatile acidity":     (0.05, 1.8),
    "citric acid":          (0.0, 1.7),
    "residual sugar":       (0.5, 70.0),
    "chlorides":            (0.005, 0.65),
    "free sulfur dioxide":  (1.0, 300.0),
    "total sulfur dioxide": (5.0, 450.0),
    "density":              (0.985, 1.04),
    "pH":                   (2.7, 4.1),
    "sulphates":            (0.2, 2.1),
    "alcohol":              (8.0, 15.0),
}

COLUNAS_NUMERICAS = list(FAIXAS.keys())

# Colunas cujos valores originais têm 3 casas decimais e que, na exportação,
# perderam o "0," e os zeros à esquerda: 0,045 -> 45 ; 0,675 -> 675.
# Para elas, um inteiro fora da faixa é dividido diretamente por 1000
# (dividir por 10 em 10 pararia cedo demais: 45 -> 0,45, que é plausível mas errado).
DIVISOR_INTEIROS = {
    "chlorides": 1000,
    "volatile acidity": 1000,
}


def _reescalar(valor, minimo, maximo, divisor_inteiro=None):
    """Corrige valores que perderam a vírgula decimal.
    - Se a coluna tem divisor fixo e o valor é inteiro, usa esse divisor.
    - Senão, divide por 10 sucessivamente até cair na faixa plausível.
    Se não cair na faixa, devolve NaN (valor irrecuperável -> imputado depois)."""
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


def _corrigir_alcohol(texto):
    """Converte o texto de alcohol em número.
    '100.333.333.333.333' -> 10.0333 ; '9.39.39.39' -> 9.3 ; '10.5' -> 10.5"""
    if pd.isna(texto):
        return np.nan
    s = str(texto).strip().replace(",", ".")
    try:
        v = float(s)
        return v
    except ValueError:
        pass
    # Padrão repetido (ex.: '9.39.39.39...'): pega o primeiro "x.y"
    partes = s.split(".")
    if len(partes) > 2 and len(set(partes[1:-1])) == 1 and len(partes[1]) == 2:
        # '9.39.39.39' -> partes = ['9','39','39',...] -> 9.3
        return float(f"{partes[0]}.{partes[1][0]}")
    # Separador de milhar: junta os dígitos e coloca a vírgula onde fica plausível
    digitos = re.sub(r"\D", "", s)
    lo, hi = FAIXAS["alcohol"]
    for k in (1, 2):
        try:
            v = float(digitos[:k] + "." + digitos[k:])
        except ValueError:
            continue
        if lo <= v <= hi:
            return v
    return np.nan


def _corrigir_tipo(texto):
    if pd.isna(texto):
        return np.nan
    s = str(texto).strip().lower()
    if s.startswith("white"):
        return "white"
    if s.startswith("red"):
        return "red"
    return np.nan


def carregar_e_limpar(caminho, remover_duplicadas=True, verbose=True):
    """Lê o CSV e devolve (df_limpo, relatorio) onde relatorio é um dict com
    a contagem de cada correção — útil para citar no relatório escrito."""
    df = pd.read_csv(caminho)
    rel = {"linhas_originais": len(df)}

    # 1) alcohol (texto -> número)
    alc_num = pd.to_numeric(df["alcohol"], errors="coerce")
    rel["alcohol_texto_corrompido"] = int((alc_num.isna() & df["alcohol"].notna()).sum())
    df["alcohol"] = df["alcohol"].apply(_corrigir_alcohol)

    # 2) colunas numéricas: força número e reescala decimais perdidos
    rel["valores_reescalados"] = {}
    rel["valores_irrecuperaveis"] = {}
    for col in COLUNAS_NUMERICAS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        lo, hi = FAIXAS[col]
        fora = df[col].notna() & ~df[col].between(lo, hi)
        div = DIVISOR_INTEIROS.get(col)
        novo = df[col].apply(lambda v: _reescalar(v, lo, hi, div))
        rel["valores_reescalados"][col] = int((fora & novo.notna()).sum())
        rel["valores_irrecuperaveis"][col] = int((fora & novo.isna()).sum())
        df[col] = novo

    # 3) type
    df["type"] = df["type"].apply(_corrigir_tipo)

    # 4) quality: alvo não se imputa -> descarta inválidos
    df["quality"] = pd.to_numeric(df["quality"], errors="coerce")
    invalida = df["quality"].isna() | ~df["quality"].between(0, 10)
    rel["linhas_quality_invalida_removidas"] = int(invalida.sum())
    df = df[~invalida].copy()
    df["quality"] = df["quality"].astype(int)

    # Linhas sem type: descarta (são raríssimas)
    sem_tipo = df["type"].isna()
    rel["linhas_sem_type_removidas"] = int(sem_tipo.sum())
    df = df[~sem_tipo].copy()

    # 5) duplicadas (evita que a mesma amostra caia em treino e teste)
    if remover_duplicadas:
        dup = df.duplicated()
        rel["duplicadas_removidas"] = int(dup.sum())
        df = df[~dup].copy()

    df = df.reset_index(drop=True)
    rel["valores_ausentes_restantes"] = {k: int(v) for k, v in df.isna().sum().items() if v}
    rel["linhas_finais"] = len(df)

    if verbose:
        print("=== Relatório de limpeza ===")
        for k, v in rel.items():
            print(f"{k}: {v}")
    return df, rel


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("uso: python limpeza.py entrada.csv [saida.csv]")
        sys.exit(1)
    dados, _ = carregar_e_limpar(sys.argv[1])
    if len(sys.argv) > 2:
        dados.to_csv(sys.argv[2], index=False)
        print(f"salvo em {sys.argv[2]}")
