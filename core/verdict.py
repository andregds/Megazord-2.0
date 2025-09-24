"""VerdictAggregator
Gera veredito local por consenso multi-timeframe com pesos e prob. mínima.
Uso:
    veredito, score, usados = VerdictAggregator.decidir(resultados, min_prob=60, pesos={"M30":2.0,...})
"""

from typing import List, Tuple

class VerdictAggregator:
    """Combina sinais por timeframe usando pesos + limiar mínimo de probabilidade."""

    @staticmethod
    def decidir(resultados: list, min_prob: float, pesos: dict) -> tuple[str, float, list]:
        """
        Retorna ('COMPRA'|'VENDA'|'AGUARDAR', score_float, lista_contribuições).
        'lista_contribuições' é [(tf, padrão, sinal, prob, peso), ...].
        """
        placar = 0.0
        usados = []
        for r in resultados:
            p = r.get("Probabilidade", 0)
            if p < min_prob:
                continue
            w = float(pesos.get(r["Timeframe"], 1.0))
            if r["Sinal"] == "COMPRA":
                placar += w
            elif r["Sinal"] == "VENDA":
                placar -= w
            usados.append((r["Timeframe"], r["Padrão"], r["Sinal"], p, w))

        if placar > 0:
            return "COMPRA", placar, usados
        if placar < 0:
            return "VENDA", placar, usados
        return "AGUARDAR", placar, usados
