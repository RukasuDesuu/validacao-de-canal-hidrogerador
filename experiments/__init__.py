"""Experimentos do artigo "MAE + SHAP" para detecção de falhas em hidrogeradores.

Fronteira dura entre computar e desenhar: os modelos escrevem scores em `cache/`,
e todo script de figura ou tabela lê exclusivamente do cache. Regenerar uma figura
nunca re-treina um modelo.
"""
