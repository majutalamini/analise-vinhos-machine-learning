"""Gera um relatório HTML a partir da auditoria e das métricas existentes."""
import hashlib
import html
import json
from datetime import datetime
from pathlib import Path

from graficos_relatorio import graficos_atuais

RAIZ = Path(__file__).resolve().parent
RESULTADOS = RAIZ / 'resultados'


def esc(valor):
    return html.escape(str(valor))


def numero(valor):
    return f'{valor:.3f}'.replace('.', ',')


def percentual(valor):
    return f'{100 * valor:.1f}%'.replace('.', ',')


def tabela(cabecalho, linhas):
    cab = ''.join(f'<th scope="col">{esc(x)}</th>' for x in cabecalho)
    corpo = ''.join('<tr>' + ''.join(f'<td>{esc(x)}</td>' for x in linha) + '</tr>' for linha in linhas)
    return f'<table><thead><tr>{cab}</tr></thead><tbody>{corpo}</tbody></table>'


def main():
    try:
        auditoria = json.loads((RESULTADOS / 'auditoria_limpeza.json').read_text(encoding='utf-8'))
        metricas = json.loads((RESULTADOS / 'metricas.json').read_text(encoding='utf-8'))
    except FileNotFoundError as erro:
        raise SystemExit(f'Arquivo ausente: {erro.filename}. Execute as etapas 1 e 2 antes do relatório.') from erro
    except (json.JSONDecodeError, UnicodeError) as erro:
        raise SystemExit(f'Não foi possível ler os resultados: {erro}') from erro

    hash_treino = metricas.get('sha256_csv_limpo')
    hash_auditoria = auditoria.get('sha256_csv_limpo')
    csv = RAIZ / 'dados/processados/vinhos_limpos.csv'
    hash_atual = hashlib.sha256(csv.read_bytes()).hexdigest() if csv.exists() else None
    avisos = []
    if not hash_treino or not hash_auditoria or hash_auditoria != hash_treino:
        avisos.append('A auditoria de limpeza e as métricas podem ser de bases diferentes. Execute novamente as etapas 1 e 2 para alinhar os resultados.')
    if not hash_atual or hash_atual != hash_treino:
        avisos.append('O CSV limpo atual está ausente ou é diferente do utilizado no treinamento. Execute novamente as etapas 1 e 2 se quiser atualizar os modelos.')
    aviso_html = ''.join(f'<div class="aviso"><strong>Atenção:</strong> {esc(a)}</div>' for a in avisos)
    imagens = {}
    if not avisos:
        try:
            imagens = graficos_atuais(metricas)
        except (FileNotFoundError, ValueError) as erro:
            avisos.append(f'Não foi possível gerar os gráficos dos modelos atuais: {erro}')
    aviso_html = ''.join(f'<div class="aviso"><strong>Atenção:</strong> {esc(a)}</div>' for a in avisos)
    def fig(*nomes):
        return ''.join(imagens[n] for n in nomes if n in imagens)

    reg = metricas['tarefas']['regressao']
    clf = metricas['tarefas']['classificacao']
    rm = reg['metricas_teste']
    cm = clf['metricas_teste']
    limpeza = [("Linhas no arquivo original", auditoria['linhas_originais']),
               ("Linhas após limpeza", auditoria['linhas_finais']),
               ("Duplicadas removidas após limpeza", auditoria['duplicadas_removidas']),
               ("Linhas removidas por nota inválida", auditoria['linhas_quality_invalida_removidas']),
               ("Linhas removidas por tipo inválido ou ausente", auditoria['linhas_sem_type_removidas'])]
    campos = [(nome, dados['ausentes no bruto'], dados['corrigidos'], dados['viraram ausentes'])
              for nome, dados in auditoria['numericas'].items()]
    classes = []
    for classe in ('baixa', 'media', 'alta'):
        d = cm['por_classe'][classe]
        classes.append(({'media': 'média'}.get(classe, classe), int(d['support']), percentual(d['precision']), percentual(d['recall']), percentual(d['f1-score'])))
    comparacao_reg = [(nome, numero(v)) for nome, v in reg['validacao_cruzada'].items()]
    comparacao_clf = [(nome, numero(v)) for nome, v in clf['validacao_cruzada'].items()]
    versoes = ', '.join(f'{k}: {v}' for k, v in metricas['versoes'].items())
    documento = f'''<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Relatório de qualidade de vinhos</title><style>
body{{font:16px/1.55 system-ui, sans-serif;color:#24222a;max-width:1050px;margin:2rem auto;padding:0 1.25rem}}
h1,h2{{color:#64283b}} h1{{margin-bottom:.2rem}} h2{{margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:.3rem}}
.sub{{color:#555}} table{{border-collapse:collapse;width:100%;margin:1rem 0}} th,td{{padding:.55rem;text-align:left;border-bottom:1px solid #ddd}} th{{background:#f5edf0}} .aviso{{background:#fff1cc;border-left:4px solid #b87500;padding:1rem;margin:1rem 0}} code{{overflow-wrap:anywhere}} body{{counter-reset:fig}} figure{{margin:1.5rem 0;break-inside:avoid;counter-increment:fig}} figcaption strong::before{{content:'Figura ' counter(fig) ' — '}} figure img{{display:block;max-width:100%;height:auto;margin:auto}} figcaption{{font-size:.92rem;color:#444;margin-top:.5rem}} @media print{{body{{max-width:none}} h2{{break-after:avoid}} tr{{break-inside:avoid}}}}
</style></head><body><h1>Relatório de qualidade de vinhos</h1>
<p class="sub">Gerado em {esc(datetime.now().strftime('%d/%m/%Y às %H:%M'))} a partir dos resultados salvos no projeto.</p>
{aviso_html}
<h2>1. Introdução e dados</h2><p>O objetivo é estimar a nota sensorial <code>quality</code> e classificar vinhos em faixas de qualidade usando onze propriedades físico-químicas e o tipo de vinho. A base do Grupo 1 contém {auditoria['linhas_originais']} registros antes da limpeza. As notas são avaliações sensoriais e não podem ser explicadas integralmente apenas pelos atributos disponíveis.</p>
<h2>2. Limpeza e pré-processamento</h2><p>A limpeza verifica o formato da base, corrige padrões conhecidos e marca como ausentes valores que não consegue corrigir com segurança.</p>
{tabela(['Indicador', 'Quantidade'], limpeza)}
<h3>Auditoria por variável numérica</h3><p>As contagens se referem à auditoria da limpeza; valores ausentes são imputados posteriormente dentro do treinamento.</p>
{tabela(['Variável', 'Ausentes no bruto', 'Corrigidos', 'Tornaram-se ausentes'], campos)}
<h2>3. Análise exploratória</h2><p>As figuras abaixo descrevem o conjunto de treino: como as notas se distribuem, como cada atributo se comporta em tintos e brancos e quais atributos se associam à nota. Concentração em poucas notas pode dificultar a previsão das faixas raras; correlação indica associação linear, não causalidade.</p>
{fig('distribuicao', 'histogramas', 'corr_nota', 'corr_matriz') or '<p>Gráficos indisponíveis: consulte os avisos acima.</p>'}
<h2>4. Treinamento e avaliação</h2><p>O fluxo separa {metricas['linhas_treino']} linhas para treino e {metricas['linhas_teste']} para teste, com semente {metricas['semente']}. A escolha é feita por validação cruzada no treino; as métricas abaixo usam o conjunto de teste.</p>
<h3>Regressão — previsão da nota</h3><p>Modelo escolhido: <strong>{esc(reg['modelo'])}</strong>. RMSE e MAE medem erros em pontos da nota; valores menores são melhores. R² compara a previsão com uma referência constante.</p>
{tabela(['Métrica no teste', 'Valor'], [('RMSE',numero(rm['rmse'])),('MAE',numero(rm['mae'])),('R²',numero(rm['r2']))])}
<p>Comparação no treino: RMSE médio na validação cruzada (menor é melhor).</p>{tabela(['Modelo','RMSE'],comparacao_reg)}
{fig('comparacao', 'previsto_real', 'imp_reg')}
<h3>Classificação — faixas de qualidade</h3><p>Modelo escolhido: <strong>{esc(clf['modelo'])}</strong>. As faixas são baixa (0–4), média (5–6) e alta (7–10). F1 macro e acurácia balanceada dão peso semelhante às classes, mesmo quando têm tamanhos diferentes.</p>
{tabela(['Métrica no teste', 'Valor'], [('F1 macro',numero(cm['f1_macro'])),('Acurácia balanceada',percentual(cm['acuracia_balanceada'])),('Acurácia',percentual(cm['acuracia']))])}
<p>Comparação no treino: F1 macro médio na validação cruzada (maior é melhor).</p>{tabela(['Modelo','F1 macro'],comparacao_clf)}
{fig('matriz_confusao', 'imp_clf')}
<h3>Resultado por classe no teste</h3><p>Precisão indica a proporção de acertos entre as previsões daquela classe; recall indica a proporção encontrada entre os casos reais da classe. F1 combina as duas medidas.</p>
{tabela(['Classe','Amostras reais','Precisão','Recall','F1'],classes)}
{fig('metricas_classe')}
<h2>5. Conclusões e limitações</h2><p>No fluxo atual, o modelo de regressão selecionado foi {esc(reg['modelo'])}, com RMSE de {numero(rm['rmse'])} no teste. Para classificação, {esc(clf['modelo'])} obteve F1 macro de {numero(cm['f1_macro'])}; o recall da classe baixa foi {percentual(cm['por_classe']['baixa']['recall'])}. As métricas por classe são importantes porque a acurácia geral pode esconder dificuldades nas classes menos frequentes.</p>
<p>O fluxo atual compara dois candidatos por tarefa e um esquema de faixas. Ele não reproduz todas as buscas de hiperparâmetros, estratégias de balanceamento e faixas presentes no notebook. Para atender integralmente ao checklist acadêmico, essas análises precisam ser revisadas e associadas à versão final escolhida.</p>
<h2>6. Referências e limites</h2><p>Os resultados descrevem esta base e esta divisão de treino e teste. As previsões não provam relações de causa e efeito. A limpeza foi criada para os problemas observados no CSV do Grupo 1.</p>
<p>Referência dos dados: Cortez et al. (2009), <em>Modeling wine preferences by data mining from physicochemical properties</em>, Decision Support Systems, 47(4), 547–553. Base: <a href="https://doi.org/10.24432/C56S3T">UCI Wine Quality</a>. Bibliotecas: <a href="https://scikit-learn.org/stable/">scikit-learn</a> e <a href="https://imbalanced-learn.org/">imbalanced-learn</a> (usada no notebook original).</p>
<p>Versões do fluxo atual: {esc(versoes)}.</p><p>SHA-256 da base limpa usada no treinamento: <code>{esc(hash_treino or 'não informado')}</code>.</p>
<p>Fontes locais: <code>resultados/auditoria_limpeza.json</code> e <code>resultados/metricas.json</code>.</p></body></html>'''
    destino = RESULTADOS / 'relatorio.html'
    destino.write_text(documento, encoding='utf-8')
    print(f'Relatório salvo em {destino}')
    for aviso in avisos:
        print(f'ATENÇÃO: {aviso}')


if __name__ == '__main__':
    main()
