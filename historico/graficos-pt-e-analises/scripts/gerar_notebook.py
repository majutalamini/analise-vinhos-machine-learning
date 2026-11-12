"""
gerar_notebook.py — Monta o notebook do trabalho a partir do template (formato "percent").

Uso:
    python gerar_notebook.py --csv group_1_winequality.csv --saida trabalho_ml_vinhos.ipynb \
        [--equipe "Nome 1, Nome 2"] [--esquema "A: 0-4 | 5-6 | 7-10"] [--textos textos.json]

- O template fica em ../templates/notebook_template.py; células começam com "# %%"
  (código) ou "# %% [markdown]" (markdown, linhas prefixadas com "# ").
- Marcadores {{NOME}} viram texto. Sem --textos, os marcadores de análise viram um aviso
  visível "[PREENCHER ...]" — a ideia é: gerar, executar, ler os resultados e gerar de novo
  com --textos contendo as análises baseadas nos números reais.
"""
import argparse
import json
import re
from pathlib import Path

import nbformat

TEMPLATE = Path(__file__).resolve().parent.parent / "templates" / "notebook_template.py"

DICAS = {
    "DIAGNOSTICO": "listar os problemas encontrados no CSV bruto, com as contagens",
    "EDA_ALVO": "comentar a distribuição de quality (concentração em 5-6, notas raras, diferença tinto × branco)",
    "EDA_HIST": "assimetrias, caudas longas (residual sugar, chlorides, SO2), bimodalidades",
    "EDA_OUTLIERS": "quais variáveis têm mais outliers e por que não removemos as linhas",
    "EDA_CORR": "variáveis mais correlacionadas com quality e pares colineares (density × alcohol, SO2 livre × total)",
    "EDA_SCATTER": "o que os scatter plots mostram sobre alcohol, volatile acidity, density",
    "REG_CV": "comparar modelos com o baseline na validação cruzada",
    "REG_TESTE": "melhor modelo no teste, RMSE/MAE/R² e o padrão dos resíduos (regressão à média)",
    "REG_DISCUSSAO": "interpretar coeficientes e importâncias; quais variáveis mais impactam",
    "CLF_DISTRIBUICAO": "desbalanceamento de cada esquema, com porcentagens",
    "CLF_COMPARACAO": "qual esquema/modelo vai melhor e por que o esquema principal foi escolhido",
    "CLF_BALANCEAMENTO": "efeito de class_weight e SMOTE no F1-macro",
    "CLF_DISCUSSAO": "confusões entre classes (quais e por quê), melhor modelo, features decisivas",
    "CONCLUSOES": "resumo final das duas partes, limitações e próximos passos",
}


def ler_celulas(texto):
    celulas, atual, tipo = [], [], None
    for linha in texto.splitlines():
        if linha.startswith("# %%"):
            if tipo is not None:
                celulas.append((tipo, atual))
            tipo = "markdown" if "[markdown]" in linha else "code"
            atual = []
        else:
            atual.append(linha)
    if tipo is not None:
        celulas.append((tipo, atual))
    saida = []
    for tipo, linhas in celulas:
        if tipo == "markdown":
            linhas = [re.sub(r"^# ?", "", l) for l in linhas]
        conteudo = "\n".join(linhas).strip("\n")
        if conteudo:
            saida.append((tipo, conteudo))
    return saida


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="group_1_winequality.csv",
                   help="nome/caminho do CSV como será usado no notebook")
    p.add_argument("--saida", default="trabalho_ml_vinhos.ipynb")
    p.add_argument("--equipe", default="(nomes da equipe)")
    p.add_argument("--esquema", default="A: 0-4 | 5-6 | 7-10",
                   help="chave do esquema de faixas usado no aprofundamento da classificação")
    p.add_argument("--textos", help="JSON {MARCADOR: texto markdown} com as análises")
    a = p.parse_args()

    textos = json.loads(Path(a.textos).read_text(encoding="utf-8")) if a.textos else {}
    valores = {"CSV": a.csv, "EQUIPE": a.equipe, "ESQUEMA_PRINCIPAL": a.esquema}
    for chave, dica in DICAS.items():
        valores[chave] = textos.get(chave, f"> ✏️ **[PREENCHER — {dica}]**")

    texto = TEMPLATE.read_text(encoding="utf-8")
    texto = re.sub(r"\{\{(\w+)\}\}", lambda m: valores.get(m.group(1), m.group(0)), texto)

    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for tipo, conteudo in ler_celulas(texto):
        nb.cells.append(nbformat.v4.new_markdown_cell(conteudo) if tipo == "markdown"
                        else nbformat.v4.new_code_cell(conteudo))
    nbformat.write(nb, a.saida)
    faltando = [k for k in DICAS if k not in textos]
    print(f"notebook salvo em {a.saida} ({len(nb.cells)} células)")
    if faltando:
        print("marcadores ainda sem texto:", ", ".join(faltando))


if __name__ == "__main__":
    main()
