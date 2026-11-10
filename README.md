# Qualidade de vinhos — Grupo 1

Trabalho final de Machine Learning em Python. A primeira parte estima a nota sensorial
`quality` por regressão. A segunda agrupa a nota em faixas e avalia classificadores.
O insumo é o [CSV do Grupo 1](dados/group_1_winequality.csv) fornecido pelo professor.

## Entrega local

O Git exclui a pasta `entrega/`. Os arquivos abaixo ficam disponíveis localmente após a preparação e a execução.

| Arquivo local | Conteúdo |
|---|---|
| `entrega/trabalho_ml_vinhos.ipynb` | Limpeza, exploração, código de modelagem, resultados e discussão |
| `entrega/group_1_winequality.csv` | Cópia byte a byte do arquivo no ZIP do professor |
| `entrega/relatorio_grupo_1.docx` | Método, auditoria, resultados, interpretação e referências |
| `entrega/apresentacao_grupo_1.pptx` | Dez slides com roteiro e duração sugerida nas notas |

A [limpeza](scripts/limpeza.py) tem uma implementação compartilhada. O [gerador do notebook](scripts/gerar_notebook.py)
combina esse código com o [template da análise](templates/notebook_template.py) e as
[discussões do Grupo 1](textos.json). O notebook abre com as saídas já incorporadas e
usa o CSV ao lado. O [gerador da documentação](scripts/gerar_documentacao.py) cria o relatório
e os slides a partir dos resultados do notebook.

## Reproduzir no PowerShell

Execute na raiz do projeto com Python 3.12. As dependências estão fixadas em
`requirements.txt`. A busca em grade pode levar alguns minutos.

```powershell
py -3.12 -m venv .venv
$pythonVinhos = ".\.venv\Scripts\python.exe"
& $pythonVinhos -m pip install -r requirements.txt

# Preparação do CSV ao lado do notebook.
New-Item -ItemType Directory -Path entrega -Force | Out-Null
Copy-Item -LiteralPath dados/group_1_winequality.csv -Destination entrega/group_1_winequality.csv

# Inspeção opcional do CSV, com auditoria na saída do terminal.
& $pythonVinhos scripts/limpeza.py entrega/group_1_winequality.csv

# Geração e execução a partir do template e das análises do Grupo 1.
& $pythonVinhos scripts/gerar_notebook.py --csv group_1_winequality.csv --saida entrega/trabalho_ml_vinhos.ipynb --equipe "Grupo 1" --textos textos.json
New-Item -ItemType Directory -Path build -Force | Out-Null
$env:VINHOS_RESULTADOS = (Resolve-Path build).Path
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
& $pythonVinhos -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 entrega/trabalho_ml_vinhos.ipynb

# O JSON e os gráficos gerados em build são a fonte da documentação.
& $pythonVinhos scripts/gerar_documentacao.py
```

O parâmetro `--csv` é interpretado pelo notebook a partir da pasta em que ele roda.
Para uma primeira execução com outro CSV, gere o notebook sem `--textos` e atualize
as discussões com os resultados obtidos antes de publicar os materiais.

## Método e fontes

- O split 80/20 usa a nota original para estratificação e semente 42. A validação cruzada
  em cinco dobras escolhe os modelos no treino. O conjunto de teste mede generalização.
- Imputação, limitação pelos percentis 1/99 e padronização são ajustadas dentro das
  dobras. A classificação compara pesos de classe e SMOTENC no treino, preservando o tipo
  do vinho como categoria durante a reamostragem.
- O script de limpeza usa os padrões observados no Grupo 1. Valores isolados sem
  reparo sustentado ficam ausentes. As médias de referência foram calculadas a partir dos
  dois arquivos da [base Wine Quality da UCI](https://doi.org/10.24432/C56S3T).
- A estrutura inicial veio do [repositório de instruções](https://github.com/majutalamini/analise-vinhos-machine-learning/tree/641c4b295166fd159406cbc2e889968c1377e980).

SHA-256 do CSV da entrega
`5c78a2700372bf3e0c7e96a7dfc4328bc56ba3a1ad033333cc53e45c26c94e68`.
