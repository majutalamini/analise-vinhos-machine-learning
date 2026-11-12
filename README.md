# Qualidade de vinhos — Grupo 1

Este projeto usa dados físico-químicos para estudar a qualidade de vinhos. O fluxo principal tem etapas separadas: `limpar_base.py` gera a base limpa e sua auditoria; `treinar.py` lê essa cópia, compara modelos e salva os dois selecionados com seu pré-processamento.

## Onde está cada coisa

| Caminho | Conteúdo |
|---|---|
| `dados/group_1_winequality.csv` | Base original, preservada sem alterações. |
| `limpeza.py` | Regras de validação, correção e auditoria da base. |
| `limpar_base.py` | Limpa o CSV original e salva uma cópia e a auditoria. |
| `treinar.py` | Lê a cópia limpa, treina, avalia e salva os modelos. |
| `dados/processados/vinhos_limpos.csv` | Cópia limpa produzida pela etapa de limpeza, antes da imputação. |
| `resultados/auditoria_limpeza.json` | Quantidade de correções, exclusões e valores ausentes. |
| `resultados/metricas.json` | Modelos comparados, seleção por validação cruzada, métricas no teste, versões e hash da base limpa usada no treinamento. |
| `modelos/regressao.joblib` | Pipeline treinado que prevê uma nota numérica. |
| `modelos/classificacao.joblib` | Pipeline treinado que prevê uma faixa de qualidade. |
| `prever.py` | Gera previsões a partir de um CSV já tratado. |
| `gerar_relatorio.py` | Monta um relatório HTML a partir da auditoria e das métricas salvas. |
| `resultados/relatorio.html` | Relatório para abrir no navegador e, se desejar, imprimir ou salvar como PDF. |
| `modelo.py` e `trabalho_ml_vinhos.ipynb` | Análise exploratória original; são estudos independentes e não geram os modelos do fluxo principal. |
| `historico/` | Material preservado da outra branch, para consulta. |

Os diretórios `dados/processados/`, `modelos/` e `resultados/` são gerados localmente e ignorados pelo Git. Executar novamente sobrescreve esses arquivos; copie-os antes se quiser guardar uma versão anterior.

## Executar com dois cliques no Windows

Abra a pasta do projeto no Explorador de Arquivos e execute, na primeira vez, nesta ordem:

1. Dê dois cliques em `1 - Limpar dados.bat`. Ele lê a base original e atualiza `dados/processados/vinhos_limpos.csv` e `resultados/auditoria_limpeza.json`; não treina modelos.
2. Dê dois cliques em `2 - Treinar modelos.bat`. Ele usa a cópia limpa já existente e atualiza os modelos e `resultados/metricas.json`; não executa a limpeza novamente. Pode levar alguns minutos.
3. Dê dois cliques em `3 - Gerar previsoes.bat` para gerar previsões de exemplo em `resultados/previsoes_exemplo.csv`.
4. Dê dois cliques em `4 - Gerar relatorio.bat` para criar e abrir `resultados/relatorio.html` no navegador. Ele usa a auditoria, as métricas e os modelos salvos para criar gráficos atuais e recupera as imagens já existentes no notebook original, identificadas separadamente; não treina novamente. No navegador, use Imprimir e escolha salvar em PDF se desejar.

Cada etapa sobrescreve seus próprios arquivos de saída. Para o relatório, as etapas 1 e 2 são necessárias; a etapa 3 é opcional. O relatório avisa se o CSV limpo ou a auditoria não corresponderem à base usada no treinamento. Se mudar a base original ou as regras de limpeza, execute a limpeza e depois treine novamente: os modelos existentes não são atualizados automaticamente. Ao concluir cada etapa, pressione uma tecla para fechar a janela. O exemplo usa a base do projeto, incluindo dados de treino, e não substitui a avaliação em `metricas.json`. Para usar outro CSV já tratado, siga as instruções de linha de comando abaixo.

Os atalhos usam o ambiente `.venv` desta instalação. Se ele não existir em outro computador, siga as instruções de instalação abaixo.

## Executar o fluxo principal

Use Python 3.13 e PowerShell na raiz do projeto:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-fluxo.txt
.\.venv\Scripts\python.exe limpar_base.py
.\.venv\Scripts\python.exe treinar.py
```

O treinamento pode levar alguns minutos. Depois, consulte `resultados/metricas.json` e os arquivos indicados acima. Os modelos são ajustados com 80% dos dados e avaliados nos 20% restantes; são salvos com o ajuste feito **apenas no treino**, para que as métricas correspondam ao modelo salvo. Para a análise exploratória antiga, use as dependências originais em `requirements.txt` e as instruções dos próprios arquivos; os resultados podem diferir porque os métodos não são idênticos.

### Gerar previsões

Depois de limpar os dados e treinar os modelos, informe um CSV **já tratado** pelas mesmas regras de `limpeza.py`, com as 12 colunas de entrada listadas abaixo. Ele pode conter `quality`, mas essa coluna não é usada na previsão. Por exemplo, para verificar o funcionamento com a própria base limpa:

```powershell
.\.venv\Scripts\python.exe prever.py dados/processados/vinhos_limpos.csv resultados/previsoes_exemplo.csv
```

O arquivo de saída contém os dados recebidos e as colunas `previsao_regressao` e `previsao_classificacao`. Este exemplo inclui dados usados no treino e serve apenas para demonstrar o comando; use as métricas do conjunto de teste para avaliar o desempenho. `prever.py` não aplica as regras de reparo de erros do CSV bruto: entradas novas precisam ser tratadas primeiro. A função de limpeza atual exige a coluna `quality` e foi criada especificamente para a base do Grupo 1; não a aplique cegamente a outra base nem a registros sem nota. Os modelos devem ser carregados apenas de arquivos confiáveis e no ambiente com as versões do treinamento.

## O que significa cada coluna

Cada linha representa uma amostra. As unidades abaixo são as convencionais da base Wine Quality da UCI; consulte a origem se uma nova base usar convenções diferentes.

| Coluna | Significado | Unidade usual |
|---|---|---|
| `fixed acidity` | Acidez fixa, principalmente ácidos não voláteis. | g/L de ácido tartárico |
| `volatile acidity` | Acidez volátil, associada principalmente ao ácido acético. | g/L de ácido acético |
| `citric acid` | Quantidade de ácido cítrico. | g/L |
| `residual sugar` | Açúcar que resta após a fermentação. | g/L |
| `chlorides` | Quantidade de cloretos, relacionada ao sal. | g/L de cloreto de sódio |
| `free sulfur dioxide` | Dióxido de enxofre livre. | mg/L |
| `total sulfur dioxide` | Dióxido de enxofre total, livre e combinado. | mg/L |
| `density` | Densidade do vinho. | g/cm³ |
| `pH` | Medida da acidez em escala logarítmica. | sem unidade |
| `sulphates` | Sulfatos presentes na amostra. | g/L de sulfato de potássio |
| `alcohol` | Teor alcoólico. | % em volume |
| `type` | Tipo do vinho: `red` (tinto) ou `white` (branco). | categoria |
| `quality` | Nota sensorial, usada como resposta a ser aprendida, não como entrada. | inteiro de 0 a 10 |

## Como a limpeza e os modelos funcionam

A limpeza valida as colunas, normaliza `type`, tenta reparar padrões conhecidos de corrupção decimal e marca como ausentes valores numéricos que não consegue corrigir com segurança. Remove notas inválidas, tipos inválidos e linhas duplicadas após o tratamento. A auditoria registra as contagens. A cópia limpa pode ter células vazias: a imputação pela mediana acontece dentro do pipeline, ajustada apenas com dados de treino ou de cada dobra, para evitar vazamento de informação.

A divisão é estratificada pela nota, com semente 42. Em cinco dobras do treino, a regressão compara Ridge e Random Forest pelo RMSE (menor é melhor). A classificação compara regressão logística e Random Forest pelo F1 macro (maior é melhor), tratando as classes de forma equilibrada. As faixas são **baixa: 0–4**, **média: 5–6**, **alta: 7–10**. Em cada pipeline, valores numéricos ausentes recebem a mediana, as variáveis numéricas são padronizadas e o tipo é codificado. O modelo escolhido é avaliado uma vez na parte de teste. RMSE e MAE indicam erros em pontos da nota; R² compara com uma referência constante. F1 macro e acurácia balanceada ajudam a observar o desempenho em classes desiguais; o relatório por classe mostra precisão e recall.

Essas previsões são estimativas baseadas nesta amostra e não estabelecem relações de causa e efeito. As regras de limpeza foram feitas para as corrupções observadas neste arquivo e devem ser revistas para novos dados.

Fonte dos dados originais: [UCI Wine Quality](https://doi.org/10.24432/C56S3T). A base do Grupo 1 foi fornecida pelo professor.
