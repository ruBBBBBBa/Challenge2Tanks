# Evaluación final (2026-11-08 18:19)

Todos contra todos determinista con las funciones de `student/league.py` (`play`, `build_table`, `head_to_head`, `series_win_prob`, `bradley_terry`): 4 mapas oficiales, 24 aleatorios (semillas 10000+) y 16 generados (`map_gen.py`, semilla 20261108), los dos lados y una partida por (mapa, lado). 1320 partidas en 2.1 min. `baseline:hard` va como referencia. **R1** (`models/eval_pc4/R1.zip`) es solo juez: se lee en su sitio, no entra en el Elo ni en ningún pool.

P(serie) = probabilidad de ganar un mejor de 3 con la regla del README (los empates se repiten en la siguiente combinación mapa/lado, luego daño, luego moneda), 2000 series simuladas.

## Ranking (Elo Bradley–Terry por conjunto de mapas; hard = referencia)

| modelo | Elo total | Elo oficiales | Elo aleatorios | Elo generados | W-D-L total | puntuación | empates | daño medio | timeouts sin daño |
|---|---|---|---|---|---|---|---|---|---|
| A-g17_8000k | 1861 | 1805 | 1862 | 1843 | 104-248-0 | 65% | 70% | 0.52 | 244 |
| A-g18_8000k | 1855 | 1753 | 1855 | 1852 | 100-250-2 | 64% | 71% | 0.51 | 244 |
| A-g19 | 1823 | 1719 | 1836 | 1804 | 68-278-6 | 59% | 79% | 0.40 | 266 |
| D2b-distill | 1765 | 1719 | 1765 | 1752 | 76-196-80 | 49% | 56% | 0.65 | 194 |
| baseline:hard | 1500 | 1500 | 1500 | 1500 | 2-88-262 | 13% | 25% | 0.23 | 72 |

## Cara a cara en mapas oficiales (fila contra columna: W-D-L · P(serie))

| | A-g17_8000k | A-g18_8000k | A-g19 | D2b-distill | baseline:hard |
|---|---|---|---|---|---|
| A-g17_8000k | · | 0-8-0 · 48% | 0-8-0 · 48% | 4-4-0 · 100% | 8-0-0 · 100% |
| A-g18_8000k | 0-8-0 · 48% | · | 0-8-0 · 48% | 0-8-0 · 48% | 6-2-0 · 100% |
| A-g19 | 0-8-0 · 48% | 0-8-0 · 48% | · | 0-8-0 · 48% | 2-6-0 · 100% |
| D2b-distill | 0-4-4 · 0% | 0-8-0 · 48% | 0-8-0 · 48% | · | 6-2-0 · 100% |
| baseline:hard | 0-0-8 · 0% | 0-2-6 · 0% | 0-6-2 · 0% | 0-2-6 · 0% | · |

## Cara a cara en mapas aleatorios (fila contra columna: W-D-L · P(serie))

| | A-g17_8000k | A-g18_8000k | A-g19 | D2b-distill | baseline:hard |
|---|---|---|---|---|---|
| A-g17_8000k | · | 0-48-0 · 50% | 0-48-0 · 50% | 24-24-0 · 100% | 32-16-0 · 100% |
| A-g18_8000k | 0-48-0 · 50% | · | 2-44-2 · 51% | 16-32-0 · 100% | 36-12-0 · 100% |
| A-g19 | 0-48-0 · 50% | 2-44-2 · 49% | · | 6-42-0 · 100% | 34-14-0 · 100% |
| D2b-distill | 0-24-24 · 0% | 0-32-16 · 0% | 0-42-6 · 0% | · | 42-6-0 · 100% |
| baseline:hard | 0-16-32 · 0% | 0-12-36 · 0% | 0-14-34 · 0% | 0-6-42 · 0% | · |

## Cara a cara en mapas generados (fila contra columna: W-D-L · P(serie))

| | A-g17_8000k | A-g18_8000k | A-g19 | D2b-distill | baseline:hard |
|---|---|---|---|---|---|
| A-g17_8000k | · | 0-32-0 · 50% | 0-32-0 · 50% | 10-22-0 · 100% | 26-6-0 · 100% |
| A-g18_8000k | 0-32-0 · 50% | · | 2-30-0 · 100% | 12-20-0 · 100% | 26-6-0 · 100% |
| A-g19 | 0-32-0 · 50% | 0-30-2 · 0% | · | 8-24-0 · 100% | 16-14-2 · 100% |
| D2b-distill | 0-22-10 · 0% | 0-20-12 · 0% | 0-24-8 · 0% | · | 28-4-0 · 100% |
| baseline:hard | 0-6-26 · 0% | 0-6-26 · 0% | 2-14-16 · 0% | 0-4-28 · 0% | · |

## Juez R1

| modelo | W-D-L oficiales | P(serie) oficiales | W-D-L aleatorios | P(serie) aleatorios | W-D-L generados | P(serie) generados |
|---|---|---|---|---|---|---|
| A-g17_8000k | 8-0-0 | 100% | 46-2-0 | 100% | 32-0-0 | 100% |
| A-g18_8000k | 8-0-0 | 100% | 42-6-0 | 100% | 30-2-0 | 100% |
| A-g19 | 8-0-0 | 100% | 44-4-0 | 100% | 28-4-0 | 100% |
| D2b-distill | 4-4-0 | 100% | 20-26-2 | 100% | 10-22-0 | 100% |
| baseline:hard | 0-0-8 | 0% | 0-8-40 | 0% | 0-0-32 | 0% |

## Conclusión

**A-g17_8000k** es el primero en Elo total (1861) y en oficiales (1805), y es el único sin ninguna derrota (104-248-0) en los tres conjuntos de mapas. Contra hard gana todas las series: 8-0-0 en oficiales, 32-16-0 en aleatorios y 26-6-0 en generados. A D2b le gana las series en los tres conjuntos.
A-g18_8000k queda casi igual en aleatorios y generados, pero en oficiales empata con D2b y gana menos a hard (6-2-0). A-g19 empata más (79 %) y remata peor a hard (2-6-0 en oficiales).
Entre los A todo son empates (P(serie) ≈ 50 %): en un cruce entre ellos se decidiría por daño o a moneda. Contra R1, los tres A ganan el 100 % de las series en todos los conjuntos de mapas; D2b, también, pero con muchos más empates.
