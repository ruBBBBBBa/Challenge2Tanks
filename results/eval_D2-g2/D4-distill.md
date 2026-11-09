# Evaluación determinista: D4-distill

Mapas: 4 oficiales y 8 aleatorios (semillas 10000+k, las de la liga), los dos lados. P(serie) = probabilidad de ganar el bo3 con la regla del torneo (`league.series_win_prob`).

## D4-distill

| rival | W-D-L oficiales | P(serie) oficiales | W-D-L aleatorios (8) | P(serie) aleatorios (8) |
|---|---|---|---|---|
| baseline:hard | 4-4-0 | 100 % | 12-4-0 | 100 % |
| A-g17_8000k | 0-8-0 | 48 % | 2-14-0 | 100 % |
| A-g15_12006k | 0-8-0 | 48 % | 0-14-2 | 0 % |
| B-g7_6000k | 2-4-2 | 46 % | 12-4-0 | 100 % |
| C-g7_4500k | 0-6-2 | 0 % | 2-12-2 | 52 % |
| R1 | 4-4-0 | 100 % | 16-0-0 | 100 % |
