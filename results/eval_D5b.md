# Evaluación final (2026-11-09 18:03)

Todos contra todos determinista con las funciones de `student/league.py` (`play`, `build_table`, `head_to_head`, `series_win_prob`, `bradley_terry`): 4 mapas oficiales, 24 aleatorios (semillas 10000+) y 16 generados (`map_gen.py`, semilla 20261108), los dos lados y una partida por (mapa, lado). 880 partidas en 3.4 min. `baseline:hard` va como referencia. **R1** (`models/reserved/R1.zip`) es solo juez: se lee en su sitio, no entra en el Elo ni en ningún pool.

P(serie) = probabilidad de ganar un mejor de 3 con la regla del README (los empates se repiten en la siguiente combinación mapa/lado, luego daño, luego moneda), 2000 series simuladas.

## Ranking (Elo Bradley–Terry por conjunto de mapas; hard = referencia)

| modelo | Elo total | Elo oficiales | Elo aleatorios | Elo generados | W-D-L total | puntuación | empates | daño medio | timeouts sin daño |
|---|---|---|---|---|---|---|---|---|---|
| A-g17_8000k | 1759 | 1802 | 1736 | 1762 | 70-194-0 | 63% | 73% | 0.51 | 192 |
| D5b-distill | 1739 | 1780 | 1720 | 1739 | 56-204-4 | 60% | 77% | 0.45 | 200 |
| D5-distill | 1724 | 1758 | 1701 | 1733 | 44-214-6 | 57% | 81% | 0.36 | 210 |
| baseline:hard | 1500 | 1500 | 1500 | 1500 | 4-96-164 | 20% | 36% | 0.25 | 86 |

## Cara a cara en mapas oficiales (fila contra columna: W-D-L · P(serie))

| | A-g17_8000k | D5b-distill | D5-distill | baseline:hard |
|---|---|---|---|---|
| A-g17_8000k | · | 0-8-0 · 48% | 0-8-0 · 48% | 8-0-0 · 100% |
| D5b-distill | 0-8-0 · 48% | · | 0-8-0 · 48% | 6-2-0 · 100% |
| D5-distill | 0-8-0 · 48% | 0-8-0 · 48% | · | 4-4-0 · 100% |
| baseline:hard | 0-0-8 · 0% | 0-2-6 · 0% | 0-4-4 · 0% | · |

## Cara a cara en mapas aleatorios (fila contra columna: W-D-L · P(serie))

| | A-g17_8000k | D5b-distill | D5-distill | baseline:hard |
|---|---|---|---|---|
| A-g17_8000k | · | 2-46-0 · 100% | 2-46-0 · 100% | 32-16-0 · 100% |
| D5b-distill | 0-46-2 · 0% | · | 0-46-2 · 0% | 32-16-0 · 100% |
| D5-distill | 0-46-2 · 0% | 2-46-0 · 100% | · | 20-26-2 · 100% |
| baseline:hard | 0-16-32 · 0% | 0-16-32 · 0% | 2-26-20 · 0% | · |

## Cara a cara en mapas generados (fila contra columna: W-D-L · P(serie))

| | A-g17_8000k | D5b-distill | D5-distill | baseline:hard |
|---|---|---|---|---|
| A-g17_8000k | · | 0-32-0 · 50% | 0-32-0 · 50% | 26-6-0 · 100% |
| D5b-distill | 0-32-0 · 50% | · | 0-32-0 · 50% | 18-14-0 · 100% |
| D5-distill | 0-32-0 · 50% | 0-32-0 · 50% | · | 18-12-2 · 100% |
| baseline:hard | 0-6-26 · 0% | 0-14-18 · 0% | 2-12-18 · 0% | · |

## Juez R1

| modelo | W-D-L oficiales | P(serie) oficiales | W-D-L aleatorios | P(serie) aleatorios | W-D-L generados | P(serie) generados |
|---|---|---|---|---|---|---|
| A-g17_8000k | 8-0-0 | 100% | 46-2-0 | 100% | 32-0-0 | 100% |
| D5b-distill | 8-0-0 | 100% | 42-4-2 | 100% | 28-4-0 | 100% |
| D5-distill | 8-0-0 | 100% | 42-6-0 | 100% | 24-8-0 | 100% |
| baseline:hard | 0-0-8 | 0% | 0-8-40 | 0% | 0-0-32 | 0% |

## Contra bots (3 semillas por mapa y lado; fuera del Elo)

| modelo | bot | W-D-L oficiales | W-D-L aleatorios | W-D-L generados | daño medio | victorias por destrucción |
|---|---|---|---|---|---|---|
| A-g17_8000k | my:bushcamper | 24-0-0 | 139-4-1 | 93-3-0 | 1.98 | 4% |
| A-g17_8000k | my:hitandhide | 24-0-0 | 139-4-1 | 92-2-2 | 2.00 | 3% |
| D5-distill | my:bushcamper | 23-1-0 | 105-11-28 | 70-10-16 | 1.84 | 11% |
| D5-distill | my:hitandhide | 20-2-2 | 104-18-22 | 74-6-16 | 1.89 | 10% |
| D5b-distill | my:bushcamper | 23-1-0 | 113-22-9 | 86-8-2 | 1.95 | 2% |
| D5b-distill | my:hitandhide | 24-0-0 | 118-16-10 | 86-5-5 | 1.93 | 4% |
