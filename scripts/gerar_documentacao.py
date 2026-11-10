"""Gera relatório e slides a partir dos resultados exportados pelo notebook."""

import argparse
import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt as WordPt, RGBColor as WordRGB
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor as SlideRGB
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt as SlidePt


TINTA = "232D38"
VINHO = "7D2347"
VERDE = "176861"
CINZA = "4D5C68"
PAPEL = "F7F4F0"
NEVE = "FFFFFF"
CLARO = "F2E5EC"
LINHA = "E7E2DD"

NOMES = {
    "fixed acidity": "Acidez fixa", "volatile acidity": "Acidez volátil",
    "citric acid": "Ácido cítrico", "residual sugar": "Açúcar residual",
    "chlorides": "Cloretos", "free sulfur dioxide": "SO₂ livre",
    "total sulfur dioxide": "SO₂ total", "density": "Densidade", "pH": "pH",
    "sulphates": "Sulfatos", "alcohol": "Álcool", "type": "Tipo",
    "type_white": "Tipo (branco vs. tinto)",
}


def numero(valor, casas=0):
    if round(valor, casas) == 0:
        valor = 0
    return f"{valor:,.{casas}f}".translate(str.maketrans(",.", ".,"))


def percentual(valor, casas=1):
    return f"{numero(100 * valor, casas)}%"


def faixa(esquema):
    return esquema.split(":", 1)[0]


def texto_simples(markdown):
    return markdown.replace("**", "").replace("`", "")


def parametros(busca):
    nomes = {
        "max_depth": "profundidade", "n_estimators": "árvores",
        "min_samples_leaf": "mínimo na folha", "max_features": "variáveis por divisão",
        "learning_rate": "taxa", "max_iter": "iterações", "max_leaf_nodes": "folhas",
        "l2_regularization": "L2", "class_weight": "peso de classe",
    }
    valores = {"None": "padrão", "passthrough": "sem reamostragem",
               "balanced": "balanceado", "balanced_subsample": "balanceado por amostra"}
    partes = []
    for chave, valor in busca["parâmetros"].items():
        chave = chave.removeprefix("modelo__")
        formatado = numero(valor, 2).rstrip("0").rstrip(",") if isinstance(valor, float) else str(valor)
        partes.append(f"{nomes.get(chave, chave)} = {valores.get(formatado, formatado)}")
    return ", ".join(partes)


def tabela_modelos(resultados, metricas, referencia):
    return [(nome, *[numero(resultados[nome][metrica], 3) for metrica in metricas])
            for nome in [referencia, *(n for n in resultados if n != referencia)]]


class Relatorio:
    def __init__(self, pasta):
        self.doc = Document()
        self.pasta = pasta
        self.figuras = 0
        self.tabelas = 0
        section = self.doc.sections[0]
        section.page_width, section.page_height = Cm(21), Cm(29.7)
        section.top_margin, section.bottom_margin = Cm(2.2), Cm(1.9)
        section.left_margin = section.right_margin = Cm(2.3)
        section.header_distance = section.footer_distance = Cm(1.0)
        section.different_first_page_header_footer = True

        normal = self.doc.styles["Normal"]
        normal.font.name, normal.font.size = "Calibri", WordPt(11)
        normal.font.color.rgb = WordRGB.from_string(TINTA)
        normal.paragraph_format.space_after = WordPt(7)
        normal.paragraph_format.line_spacing = 1.12
        for name, size, cor in [("Title", 40, VINHO), ("Heading 1", 25, VINHO),
                                ("Heading 2", 14, TINTA)]:
            style = self.doc.styles[name]
            style.font.name, style.font.size = "Calibri", WordPt(size)
            style.font.color.rgb = WordRGB.from_string(cor)
            style.paragraph_format.space_before = WordPt(9)
            style.paragraph_format.space_after = WordPt(6)

        cabecalho = section.header.paragraphs[0]
        cabecalho.add_run("MACHINE LEARNING    /    GRUPO 1").bold = True
        cabecalho.runs[0].font.size = WordPt(8)
        cabecalho.runs[0].font.color.rgb = WordRGB.from_string(VINHO)
        rodape = section.footer.paragraphs[0]
        rodape.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        rodape.add_run("QUALIDADE DE VINHOS    /    ").font.size = WordPt(8)
        campo = OxmlElement("w:fldSimple")
        campo.set(qn("w:instr"), "PAGE")
        rodape._p.append(campo)
        self.doc.core_properties.title = "Qualidade de vinhos"
        self.doc.core_properties.author = "Grupo 1"
        self.doc.core_properties.subject = "Trabalho final de Machine Learning"

    def pagina(self, titulo):
        self.doc.add_page_break()
        self.doc.add_heading(titulo, 1)

    def p(self, texto):
        for paragrafo in texto_simples(texto).split("\n\n"):
            self.doc.add_paragraph(paragrafo)

    def h(self, titulo):
        self.doc.add_heading(titulo, 2)

    def legenda(self, texto):
        p = self.doc.add_paragraph()
        p.paragraph_format.space_after = WordPt(4)
        run = p.add_run(texto)
        run.bold, run.font.size = True, WordPt(9)
        run.font.color.rgb = WordRGB.from_string(CINZA)
        return p

    def tabela(self, titulo, colunas, dados, larguras=None):
        self.tabelas += 1
        self.legenda(f"Tabela {self.tabelas} · {titulo}").paragraph_format.keep_with_next = True
        tabela = self.doc.add_table(rows=0, cols=len(colunas))
        tabela.alignment = WD_TABLE_ALIGNMENT.CENTER
        tabela.autofit = False
        larguras = larguras or [16.4 / len(colunas)] * len(colunas)
        for coluna, largura in zip(tabela.columns, larguras):
            coluna.width = Cm(largura)
        for i, valores in enumerate([colunas, *dados]):
            linha = tabela.add_row()
            for celula, valor, largura in zip(linha.cells, valores, larguras):
                celula.width = Cm(largura)
                celula.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                fundo = OxmlElement("w:shd")
                fundo.set(qn("w:fill"), VINHO if i == 0 else PAPEL if i % 2 else NEVE)
                celula._tc.get_or_add_tcPr().append(fundo)
                p = celula.paragraphs[0]
                p.paragraph_format.space_before = WordPt(3)
                p.paragraph_format.space_after = WordPt(3)
                run = p.add_run(str(valor))
                run.bold = i == 0
                run.font.size = WordPt(9)
                run.font.color.rgb = WordRGB.from_string(NEVE if i == 0 else TINTA)
        self.doc.add_paragraph().paragraph_format.space_after = WordPt(0)

    def figura(self, arquivo, legenda, largura=16.0, altura=10.0):
        caminho = self.pasta / f"{arquivo}.png"
        with Image.open(caminho) as imagem:
            largura = min(largura, altura * imagem.width / imagem.height)
        self.doc.add_picture(str(caminho), width=Cm(largura))
        paragrafo = self.doc.paragraphs[-1]
        paragrafo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragrafo.paragraph_format.keep_with_next = True
        self.figuras += 1
        self.legenda(f"Figura {self.figuras} · {legenda}")


def gerar_relatorio(saida, pasta, r, a):
    rel = Relatorio(pasta)
    l, reg, clf = r["limpeza"], r["regressao"], r["classificacao"]
    p = rel.doc.add_paragraph()
    p.paragraph_format.space_before = WordPt(102)
    run = p.add_run("TRABALHO FINAL DE MACHINE LEARNING")
    run.font.size, run.font.color.rgb = WordPt(12), WordRGB.from_string(VERDE)
    run.bold = True
    rel.doc.add_heading("Qualidade de\nvinhos", 0)
    rel.h("Predição da nota e classificação por faixas")
    rel.p("Grupo 1")
    rel.p("Wine Quality Dataset · Dados fornecidos pelo professor")
    rel.p("Novembro de 2026")

    rel.pagina("1. Problema e dados")
    rel.p("O objetivo é estimar a nota sensorial de vinhos tintos e brancos a partir de medidas "
          "físico-químicas. A regressão prevê um valor contínuo na escala da nota. A classificação "
          "prevê a faixa sensorial à qual a nota pertence.")
    rel.h("Origem e variáveis")
    rel.p("A base Wine Quality reúne variantes tinto e branco de vinho verde português (Cortez et al., "
          "2009). Usamos a versão fornecida pelo professor ao Grupo 1. Ela contém 11 atributos "
          "físico-químicos (Tabela 1), o tipo e o alvo quality, uma nota inteira de 0 a 10. Os CSVs "
          "originais da UCI serviram para conferir médias agregadas.")
    rel.tabela("Atributos e papel na análise", ["Grupo", "Variáveis"], [
        ("Acidez", "Acidez fixa, acidez volátil, ácido cítrico e pH"),
        ("Composição", "Açúcar residual, cloretos, sulfatos, SO₂ livre e SO₂ total"),
        ("Propriedades físicas", "Densidade e teor alcoólico"),
        ("Categoria e alvo", "Tipo de vinho (atributo) e quality (alvo)"),
    ], [5.1, 11.3])
    rel.h("Unidade de análise e partição")
    rel.p(f"Após a limpeza, ficaram {numero(l['linhas_finais'])} registros, sendo "
          f"{numero(r['tipos']['white'])} brancos e {numero(r['tipos']['red'])} tintos. "
          f"Dividimos as observações em {numero(r['particao']['treino'])} para treino e "
          f"{numero(r['particao']['teste'])} para teste. A divisão é compartilhada entre as tarefas "
          "e estratificada pela nota original para manter proporções semelhantes. A Figura 1 mostra "
          "a distribuição no treino.")
    rel.figura("alvo", "Notas e tipos de vinho no conjunto de treino.", altura=5.2)

    rel.pagina("2. Limpeza e pré-processamento")
    rel.p(a["DIAGNOSTICO"])
    rel.tabela("Auditoria por variável antes e depois da limpeza",
               ["Variável", "Ausentes no bruto", "Corrigidos", "Viraram ausentes", "Ausentes ao final"],
               [(NOMES[nome], *(numero(v) for v in contagens.values()))
                for nome, contagens in r["tabela_limpeza"].items()], [4.0, 3.0, 2.5, 3.4, 3.5])
    rel.p("Na Tabela 2, as três primeiras colunas de contagem se referem ao CSV bruto. A última considera "
          "as exclusões e a deduplicação. Cloretos e acidez volátil seguem a regra de divisão por "
          "1.000 para inteiros fora da faixa. Densidade segue um ajuste por potências de 10. "
          "Valores isolados sem reparo sustentado foram marcados como ausentes.")
    rel.tabela("Conferência de médias agregadas, Grupo 1 / UCI",
               ["Tipo", "Cloretos", "Acidez volátil", "Densidade"],
               [("Tinto" if tipo == "red" else "Branco",
                 *(f"{numero(r['medias_grupo'][tipo][col], 5)} / {numero(val, 5)}"
                   for col, val in uci.items())) for tipo, uci in r["uci"].items()],
               [3.1, 4.4, 4.5, 4.4])
    rel.p("A Tabela 3 confere a plausibilidade da escala por tipo com médias agregadas. Pequenas "
          "diferenças são compatíveis com os reparos, as exclusões e a deduplicação.")
    rel.h("Tratamento ajustado dentro de cada dobra")
    rel.p("Imputamos as variáveis numéricas pela mediana, limitamos os extremos aos percentis 1 e "
          "99 e padronizamos. Codificamos o tipo por one-hot, com tinto como referência. Todos os "
          "parâmetros são aprendidos nas dobras de treino. Na classificação, a reamostragem "
          "opcional recebe o tipo como categoria antes do one-hot.")

    rel.pagina("3. Análise exploratória")
    rel.p(a["EDA_ALVO"])
    rel.p(a["EDA_HIST"])
    rel.p(a["EDA_OUTLIERS"])
    rel.figura("correlacao", "Correlações de Pearson calculadas no treino.", altura=10.4)
    rel.p(a["EDA_CORR"])
    rel.p(a["EDA_SCATTER"])
    rel.p("A Figura 2 resume associações marginais, sensíveis à composição da amostra. "
          "Os histogramas, boxplots e gráficos de dispersão completos estão no notebook.")

    rel.pagina("4. Regressão · Avaliação")
    rel.p("A referência prevê a nota média do treino. A regressão linear permite ler os "
          "coeficientes, a Ridge adiciona regularização e Random Forest e HistGradientBoosting "
          "capturam relações não lineares. O RMSE enfatiza erros grandes, o MAE expressa o erro "
          "absoluto na escala da nota e o R² compara a variância residual com a variância do alvo.")
    rel.p(a["REG_CV"])
    rel.tabela("Melhores configurações por RMSE em cinco dobras",
               ["Modelo", "RMSE CV", "Melhores parâmetros"],
               [(n, numero(v["RMSE CV"], 3), parametros(v)) for n, v in reg["buscas"].items()],
               [4.4, 2.1, 9.9])
    rel.p("A Tabela 4 mostra os melhores parâmetros. A grade Ridge avaliou α de 0,01 a 100 "
          "(6 valores). Random Forest cruzou "
          "200/400 árvores, profundidade livre/20, folha mínima 1/3 e sqrt/50% das variáveis. "
          "HistGradientBoosting cruzou taxa 0,03/0,10, 200/500 iterações, 15/31 folhas e L2 de 0/1.")
    rel.tabela("Desempenho no conjunto de teste", ["Modelo", "RMSE", "MAE", "R²"],
               tabela_modelos(reg["teste"], ["RMSE", "MAE", "R²"], "Baseline (média)"),
               [7.4, 3.0, 3.0, 3.0])
    rel.p("A Tabela 5 compara os candidatos e a referência no teste. " + a["REG_TESTE"])

    rel.pagina("5. Regressão · Interpretação")
    rel.figura("regressao_residuos", "Notas previstas por nota real e distribuição dos resíduos no teste.",
               altura=6.0)
    rel.p("A Figura 3 mostra o comportamento das previsões por nota. " + a["REG_DISCUSSAO"])
    rel.figura("regressao_importancia", "Queda no R² após embaralhar cada variável no teste.",
               altura=8.5)
    rel.p("A importância por permutação da Figura 4 descreve quanto o modelo ajustado depende de cada variável "
          "neste teste. Variáveis correlacionadas podem carregar parte da mesma informação, e "
          "os coeficientes descrevem associações condicionais.")

    rel.pagina("6. Classificação · Faixas e validação")
    rel.p(a["CLF_DISTRIBUICAO"])
    rel.p("A média dos F1 das classes atribui peso igual a cada classe. Usamos F1-macro na seleção "
          "e a acurácia balanceada, média da revocação por classe, como leitura complementar.")
    esquemas = list(clf["distribuicoes"])
    candidatos = list(dict.fromkeys(linha["Modelo"] for linha in clf["cv"]))
    cv = {(linha["Modelo"], linha["Esquema"]): linha["F1-macro"] for linha in clf["cv"]}
    rel.tabela("F1-macro médio por faixa e modelo em cinco dobras",
               ["Modelo", *(faixa(e) for e in esquemas)],
               [(n, *(numero(cv[n, e], 3) for e in esquemas)) for n in candidatos],
               [7.4, 3.0, 3.0, 3.0])
    rel.p("A Tabela 6 compara as três tarefas de classificação. " + a["CLF_COMPARACAO"])
    rel.tabela("Balanceamento com Random Forest no esquema A",
               ["Estratégia", "F1-macro CV", "Acurácia balanceada"],
               [(n, numero(v["F1-macro"], 3), numero(v["Acur. balanceada"], 3))
                for n, v in clf["balanceamento"].items()], [7.4, 4.4, 4.6])
    rel.p("A Tabela 7 isola o efeito do balanceamento na floresta. " + a["CLF_BALANCEAMENTO"])
    rel.p("Na busca final, variamos C e peso de classe na logística, C e gamma na SVM, "
          "profundidade e tamanho mínimo da folha na floresta e taxa, iterações e folhas no "
          "boosting. A logística e a floresta também testaram SMOTENC como opção da grade.")

    rel.pagina("7. Classificação · Resultado no teste")
    rel.tabela("Melhores configurações por F1-macro em cinco dobras",
               ["Modelo", "F1-macro CV", "Melhores parâmetros"],
               [(n, numero(v["F1-macro CV"], 3), parametros(v)) for n, v in clf["buscas"].items()],
               [4.3, 2.5, 9.6])
    rel.tabela("Resultados no conjunto de teste, esquema A",
               ["Modelo", "F1-macro", "Acur. balanceada", "Acurácia"],
               tabela_modelos(clf["teste"], ["F1-macro", "Acur. balanceada", "Acurácia"],
                              "Baseline (majoritária)"), [7.0, 2.9, 3.7, 2.8])
    rel.figura("classificacao_confusao", "Matriz de confusão em contagens e percentuais por classe.",
               altura=5.9)
    rel.p("As Tabelas 8 e 9 separam seleção e teste. A Figura 5 mostra os erros por faixa. "
          + a["CLF_DISCUSSAO"])

    rel.pagina("8. Conclusões e referências")
    rel.p(a["CONCLUSOES"])
    rel.h("Limitações")
    rel.p("A nota sensorial tem componente subjetivo, e os dados não incluem preço, safra ou "
          "variedade da uva. Os poucos registros nas faixas extremas limitam a precisão das "
          "estimativas por classe. Os resultados descrevem esta amostra e esta partição; "
          "a generalização a outros vinhos exige validação externa.")
    rel.h("Próximas investigações")
    rel.p("Testar uma abordagem ordinal, que preserve a ordem das faixas, e avaliar modelos "
          "separados para tintos e brancos. Uma amostra maior de notas extremas permitiria "
          "verificar se os padrões de erro se repetem.")
    rel.h("Referências")
    rel.p("Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine "
          "preferences by data mining from physicochemical properties. Decision Support Systems, "
          "47(4), 547–553. https://doi.org/10.1016/j.dss.2009.05.016")
    rel.p("Cortez, P. et al. (2009). Wine Quality [dataset]. UCI Machine Learning Repository. "
          "https://doi.org/10.24432/C56S3T")
    rel.p("Scikit-learn developers. Scikit-learn User Guide. https://scikit-learn.org/stable/user_guide.html")
    rel.p("Imbalanced-learn developers. SMOTENC. https://imbalanced-learn.org/stable/references/"
          "generated/imblearn.over_sampling.SMOTENC.html")
    rel.doc.save(saida)


class Slides:
    def __init__(self, pasta):
        self.pasta = pasta
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(13.333), Inches(7.5)
        self.prs.core_properties.title = "Qualidade de vinhos"
        self.prs.core_properties.author = "Grupo 1"

    def retangulo(self, x, y, w, h, cor):
        caixa = self.slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, *map(Inches, (x, y, w, h)))
        caixa.fill.solid()
        caixa.fill.fore_color.rgb = SlideRGB.from_string(cor)
        caixa.line.fill.background()

    def texto(self, x, y, w, h, valor, tamanho=20, cor=TINTA, negrito=False, alinhar=PP_ALIGN.LEFT):
        caixa = self.slide.shapes.add_textbox(*map(Inches, (x, y, w, h)))
        tf = caixa.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(0)
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = alinhar
        p.line_spacing = 1.04
        run = p.add_run()
        run.text = str(valor)
        run.font.name, run.font.size = "Calibri", SlidePt(tamanho)
        run.font.color.rgb, run.font.bold = SlideRGB.from_string(cor), negrito

    def pagina(self, secao, titulo, tempo, roteiro, subtitulo=""):
        self.slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.slide.background.fill.solid()
        self.slide.background.fill.fore_color.rgb = SlideRGB.from_string(PAPEL)
        self.retangulo(0, 0, 13.333, 0.11, VINHO)
        self.texto(0.68, 0.33, 12, 0.27, secao.upper(), 11, VINHO, True)
        self.texto(0.68, 0.72, 12, 0.74, titulo, 38, TINTA, True)
        self.texto(0.70, 1.51, 11.95, 0.40, subtitulo, 17, CINZA)
        self.retangulo(0.68, 6.96, 11.98, 0.014, LINHA)
        self.texto(0.68, 7.02, 8, 0.25, "GRUPO 1  /  MACHINE LEARNING", 10, CINZA, True)
        self.texto(11.3, 7.02, 1.35, 0.25, f"{len(self.prs.slides):02d} / 10", 10, CINZA, True,
                   PP_ALIGN.RIGHT)
        self.slide.notes_slide.notes_text_frame.text = f"{tempo}\n\n{roteiro}"

    def imagem(self, nome, x, y, w, h):
        caminho = self.pasta / f"{nome}.png"
        with Image.open(caminho) as img:
            escala = min(w / img.width, h / img.height)
            largura, altura = img.width * escala, img.height * escala
        self.slide.shapes.add_picture(str(caminho), Inches(x + (w - largura) / 2),
                                      Inches(y + (h - altura) / 2), Inches(largura), Inches(altura))

    def numero(self, x, y, w, valor, legenda):
        self.retangulo(x, y, w, 1.13, CLARO)
        self.texto(x + .20, y + .10, w - .40, .53, valor, 29, VINHO, True)
        self.texto(x + .20, y + .64, w - .40, .32, legenda, 14, TINTA)

    def tabela(self, x, y, larguras, cabecalho, linhas, destaque=None, altura=.55):
        for i, linha in enumerate([cabecalho, *linhas]):
            linha_y = y + i * altura
            cor = VINHO if i == 0 else CLARO if linha[0] == destaque else NEVE if i % 2 else PAPEL
            self.retangulo(x, linha_y, sum(larguras), altura, cor)
            coluna_x = x
            for j, (valor, largura) in enumerate(zip(linha, larguras)):
                self.texto(coluna_x + .12, linha_y, largura - .24, altura, valor, 15,
                           NEVE if i == 0 else TINTA, i == 0 or linha[0] == destaque,
                           PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER)
                coluna_x += largura


def gerar_slides(saida, pasta, r, a):
    s = Slides(pasta)
    l, reg, clf = r["limpeza"], r["regressao"], r["classificacao"]
    nr, nc = reg["selecionado"], clf["selecionado"]
    mr, mc = reg["teste"][nr], clf["teste"][nc]
    tempo = ["0:15", "0:45", "1:15", "1:00", "0:45", "1:15", "1:00", "1:00", "1:15", "0:30"]
    s.pagina("Trabalho final", "Qualidade de vinhos", tempo[0],
             "Apresente o Grupo 1. Explique que usamos medidas físico-químicas para estimar a "
             "nota sensorial de vinhos e prever a faixa de qualidade.",
             "Regressão da nota sensorial e classificação por faixas")
    s.retangulo(.68, 2.28, 11.97, 3.75, VINHO)
    s.texto(1.05, 2.67, 10.8, .40, "WINE QUALITY DATASET", 17, "F4D7E4", True)
    s.texto(1.04, 3.18, 10.8, 1.02, "Da composição à avaliação sensorial", 36, NEVE, True)
    s.texto(1.06, 4.85, 10.4, .46, "Grupo 1  ·  Novembro de 2026", 20, NEVE)

    s.pagina("01 / Dados", "O que estamos tentando prever?", tempo[1],
             f"São 11 medidas físico-químicas e o tipo, usados para prever uma nota de 0 a 10. "
             f"O arquivo bruto tem {numero(l['linhas_originais'])} linhas. Explique que a divisão "
             "80/20 preserva a proporção das notas e é compartilhada pelas duas tarefas.",
             "O alvo quality é uma nota inteira de 0 a 10 atribuída por degustadores.")
    s.numero(.69, 2.18, 3.74, numero(l["linhas_originais"]), "linhas no CSV do professor")
    s.numero(4.79, 2.18, 3.74, numero(r["particao"]["treino"]), "registros para treino")
    s.numero(8.90, 2.18, 3.74, numero(r["particao"]["teste"]), "registros para teste")
    s.imagem("alvo", .83, 3.57, 11.6, 3.10)

    s.pagina("02 / Limpeza", "Recuperar a escala das medidas", tempo[2],
             texto_simples(a["DIAGNOSTICO"]) + " Explique que o pH 87,48 virou ausente, "
             "e que as médias da UCI foram usadas como referência agregada.",
             f"{numero(l['linhas_originais'])} linhas brutas  →  {numero(l['linhas_finais'])} linhas para análise")
    s.tabela(.69, 2.24, [3.1, 2.6, 2.6, 3.65], ["Variável", "Exemplo bruto", "Após reparo", "Valores reparados"],
             [("Cloretos", "76", "0,076", numero(l["numericas"]["chlorides"]["corrigidos"])),
              ("Densidade", "9,978", "0,9978", numero(l["numericas"]["density"]["corrigidos"])),
              ("Álcool", "100.333.333…", "10,033…", numero(l["numericas"]["alcohol"]["corrigidos"])),
              ("Acidez volátil", "675", "0,675", numero(l["numericas"]["volatile acidity"]["corrigidos"]))])
    s.texto(.72, 5.47, 11.8, .59,
            f"{numero(l['duplicadas_removidas'])} duplicatas · 2 notas inválidas · 1 linha sem tipo",
            20, VERDE, True)

    s.pagina("03 / Exploração", "Álcool e densidade se destacam", tempo[3],
             texto_simples(a["EDA_CORR"]) + " A concentração nas notas centrais ajuda a entender "
             "por que os extremos são mais difíceis.",
             "Correlações de Pearson com a nota no conjunto de treino.")
    melhores = sorted(((k, v) for k, v in r["correlacoes"]["quality"].items() if k != "quality"),
                      key=lambda item: abs(item[1]), reverse=True)[:4]
    for i, (nome, val) in enumerate(melhores):
        y = 2.27 + i * .91
        s.texto(.71, y, 3.1, .43, NOMES[nome], 22, TINTA, True)
        s.retangulo(3.87, y + .06, 6.39, .33, LINHA)
        s.retangulo(3.87, y + .06, 6.39 * abs(val) / .60, .33, VERDE if val > 0 else VINHO)
        s.texto(10.43, y, 1.70, .43, f"{'+' if val > 0 else '−'}{numero(abs(val), 2)}",
                22, TINTA, True, PP_ALIGN.RIGHT)
    s.texto(.72, 6.17, 11.8, .35, "Verde = associação positiva · Vinho = associação negativa", 14, CINZA)

    s.pagina("04 / Método", "O pipeline aprende apenas no treino", tempo[4],
             "Em cada dobra, imputamos pela mediana, limitamos entre P1 e P99 e padronizamos. "
             "O tipo entra por one-hot. Explique por que o SMOTENC é opcional e vem antes do "
             "one-hot, sempre na parte de treino da dobra. O teste permanece reservado.",
             "Validação cruzada de cinco dobras no treino · semente 42")
    etapas = [("01", "Imputar", "Mediana"), ("02", "Limitar extremos", "Percentis 1 e 99"),
              ("03", "Padronizar", "Média 0 e desvio 1"), ("04", "Ajustar", "Tipo e modelo")]
    for i, (seq, titulo, complemento) in enumerate(etapas):
        x = .70 + i * 3.06
        s.retangulo(x, 2.65, 2.76, 1.82, NEVE)
        s.texto(x + .17, 2.84, 2.4, .32, seq, 14, VINHO, True)
        s.texto(x + .17, 3.24, 2.4, .48, titulo, 19, TINTA, True)
        s.texto(x + .17, 3.82, 2.4, .36, complemento, 15, CINZA)
    s.texto(.72, 5.23, 11.8, .56,
            "Classificação com SMOTENC: reamostrar no treino → codificar o tipo → ajustar o modelo",
            20, VERDE, True)

    s.pagina("05 / Regressão", "Quanto erramos ao prever a nota?", tempo[5],
             texto_simples(a["REG_TESTE"]) + " Explique por que RMSE é o critério de seleção "
             "e o que MAE e R² acrescentam à leitura.",
             f"{nr} selecionado por RMSE na validação cruzada.")
    s.tabela(.70, 2.26, [5.3, 2.1, 2.1, 2.45], ["Modelo", "RMSE", "MAE", "R²"],
             tabela_modelos(reg["teste"], ["RMSE", "MAE", "R²"], "Baseline (média)"), nr)
    s.texto(.72, 6.05, 11.8, .42, f"RMSE no teste · {numero(mr['RMSE'], 3)} na escala da nota",
            21, VERDE, True)

    s.pagina("06 / Regressão", "O modelo aproxima os extremos do centro", tempo[6],
             texto_simples(a["REG_DISCUSSAO"]) + " Explique que resíduos positivos indicam "
             "subestimação e negativos indicam superestimação.", "Valores previstos por nota real e resíduos.")
    s.imagem("regressao_residuos", .70, 2.09, 11.92, 3.88)
    principal, impacto = max(reg["importancias"].items(), key=lambda item: item[1])
    s.texto(.72, 6.19, 11.8, .40,
            f"Embaralhar {NOMES[principal].lower()} reduz o R² em {numero(impacto, 3)} no teste.",
            19, VERDE, True)

    s.pagina("07 / Classificação", "Três maneiras de definir as faixas", tempo[7],
             texto_simples(a["CLF_COMPARACAO"]) + " Justifique F1-macro pela necessidade de dar "
             "o mesmo peso a todas as faixas. Cite a comparação de balanceamento: "
             + texto_simples(a["CLF_BALANCEAMENTO"]),
             "O esquema A é o foco da avaliação final.")
    s.imagem("classificacao_faixas", .72, 2.08, 11.88, 3.77)
    menor = min(clf["distribuicoes"][clf["esquema"]].values())
    s.texto(.72, 6.12, 11.8, .45,
            f"No esquema A, a menor classe representa {percentual(menor / r['particao']['treino'])} do treino.",
            20, VERDE, True)

    s.pagina("08 / Classificação", "Em quais faixas o modelo mais erra?", tempo[8],
             texto_simples(a["CLF_DISCUSSAO"]) + " Mostre a diferença entre contagens e recall "
             "por classe na matriz normalizada.",
             f"{nc} selecionado por F1-macro na validação cruzada.")
    s.imagem("classificacao_confusao", .68, 2.00, 11.97, 3.46)
    s.numero(.69, 5.64, 3.75, numero(mc["F1-macro"], 3), "F1-macro no teste")
    s.numero(4.79, 5.64, 3.75, numero(mc["Acur. balanceada"], 3), "acurácia balanceada")
    s.numero(8.89, 5.64, 3.75, percentual(clf["relatorio"]["baixa"]["recall"]), "revocação da classe baixa")

    s.pagina("09 / Conclusão", "O que os resultados mostram", tempo[9],
             f"Recapitule {nr} na regressão e {nc} na classificação. "
             "Destaque a dificuldade nas notas extremas e encerre com a proposta de validação em nova amostra.",
             "As medidas físico-químicas capturam parte da variação da nota sensorial.")
    s.numero(.70, 2.22, 5.70, f"RMSE {numero(mr['RMSE'], 3)}", f"Regressão · {nr}")
    s.numero(6.73, 2.22, 5.90, f"F1-macro {numero(mc['F1-macro'], 3)}", f"Classificação · {nc}")
    s.texto(.73, 3.90, 11.8, .50, "Poucos vinhos nas notas extremas limitam as estimativas.", 23, TINTA, True)
    s.texto(.73, 4.67, 11.8, .52, "A nota é sensorial e a amostra não inclui safra, uva ou preço.", 21, CINZA)
    s.texto(.73, 5.49, 11.8, .58, "Próximo passo: validar em outra amostra e testar classificação ordinal.",
            20, VERDE, True)
    s.prs.save(saida)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resultados", type=Path, default=Path("build"))
    parser.add_argument("--textos", type=Path, default=Path("textos.json"))
    parser.add_argument("--saida", type=Path, default=Path("entrega"))
    args = parser.parse_args()
    r = json.loads((args.resultados / "resultados.json").read_text(encoding="utf-8"))
    a = json.loads(args.textos.read_text(encoding="utf-8"))
    args.saida.mkdir(parents=True, exist_ok=True)
    gerar_relatorio(args.saida / "relatorio_grupo_1.docx", args.resultados, r, a)
    gerar_slides(args.saida / "apresentacao_grupo_1.pptx", args.resultados, r, a)


if __name__ == "__main__":
    main()
