# Evaluación determinista: D2-distill

Mapas: 4 oficiales y 8 aleatorios (semillas 10000+k, las de la liga), los dos lados. P(serie) = probabilidad de ganar el bo3 con la regla del torneo (`league.series_win_prob`).

## D2-distill

| rival | W-D-L oficiales | P(serie) oficiales | W-D-L aleatorios (8) | P(serie) aleatorios (8) |
|---|---|---|---|---|
| baseline:hard | 2-6-0 | 100 % | 8-4-4 | 80 % |
| A-g7_8000k | 0-8-0 | 48 % | 0-16-0 | 49 % |
| A-g6_4000k | 0-8-0 | 48 % | 0-16-0 | 49 % |
| A-g5_24000k | 0-6-2 | 0 % | 0-16-0 | 49 % |
| B-g4_3000k | 0-8-0 | 48 % | 0-14-2 | 0 % |
| C-g4_6000k | 2-6-0 | 100 % | 0-16-0 | 49 % |
| E-g1_2000k | 0-8-0 | 48 % | 0-14-2 | 0 % |
| R1 | 0-8-0 | 48 % | 0-14-2 | 0 % |
