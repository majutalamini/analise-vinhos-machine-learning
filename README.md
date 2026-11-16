# Qualidade de vinhos — Grupo 1

Trabalho final de Machine Learning em Python. A primeira parte estima a nota sensorial
`quality` por regressão. A segunda agrupa a nota em faixas e avalia classificadores.
O insumo é o [CSV do Grupo 1](dados/group_1_winequality.csv) fornecido pelo professor.

## Código do trabalho

O [modelo.py](modelo.py) é o script de Machine Learning. Ele carrega os dados, prepara
os atributos, treina e compara os modelos e avalia os resultados.
O [notebook](trabalho_ml_vinhos.ipynb) apresenta esse código por etapas, com explicações,
gráficos e resultados executados. Ambos usam [limpeza.py](limpeza.py) para tratar
o [CSV do Grupo 1](dados/group_1_winequality.csv).

| Arquivo | Conteúdo |
|---|---|
| `modelo.py` | Script com exploração, treinamento e avaliação dos modelos |
| `trabalho_ml_vinhos.ipynb` | Análise exploratória, pipelines, regressão, classificação e discussão |
| `limpeza.py` | Leitura, validação, reparos e auditoria dos dados |
| `dados/group_1_winequality.csv` | CSV original fornecido pelo professor |
| `requirements.txt` | Dependências para executar o trabalho |

O notebook abre com as saídas já incorporadas. O relatório e a apresentação estão
na pasta local `entrega/`.

## Reproduzir no PowerShell

Execute na raiz do projeto com Python 3.12. As dependências estão fixadas em
`requirements.txt`. A busca em grade pode levar alguns minutos.

```powershell
py -3.12 -m venv .venv
$pythonVinhos = ".\.venv\Scripts\python.exe"
& $pythonVinhos -m pip install -r requirements.txt

# Inspeção opcional do CSV, com auditoria na saída do terminal.
& $pythonVinhos limpeza.py dados/group_1_winequality.csv

# Execução do script de Machine Learning.
& $pythonVinhos modelo.py

# Execução de todas as células do notebook.
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
& $pythonVinhos -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 trabalho_ml_vinhos.ipynb
```

O caminho do CSV está na variável `CAMINHO_CSV` do notebook e é relativo à pasta
do projeto. Ao usar outra base, revise a limpeza e as discussões com os novos resultados.

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
