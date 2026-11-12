"""Limpa o CSV original e salva uma cópia e a auditoria."""
import hashlib
import json
from pathlib import Path

from limpeza import carregar_e_limpar

RAIZ = Path(__file__).resolve().parent
ORIGINAL = RAIZ / 'dados/group_1_winequality.csv'
LIMPO = RAIZ / 'dados/processados/vinhos_limpos.csv'
AUDITORIA = RAIZ / 'resultados/auditoria_limpeza.json'


def main():
    dados, auditoria = carregar_e_limpar(ORIGINAL, verbose=False)
    LIMPO.parent.mkdir(parents=True, exist_ok=True)
    AUDITORIA.parent.mkdir(parents=True, exist_ok=True)
    dados.to_csv(LIMPO, index=False)
    auditoria['sha256_csv_original'] = hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()
    auditoria['sha256_csv_limpo'] = hashlib.sha256(LIMPO.read_bytes()).hexdigest()
    AUDITORIA.write_text(json.dumps(auditoria, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Base limpa salva em {LIMPO} ({len(dados)} linhas).')
    print(f'Auditoria salva em {AUDITORIA}.')


if __name__ == '__main__':
    main()
