# Evaluación determinista: D3a-distill, D2b-distill

Mapas: 4 oficiales y 8 aleatorios (semillas 10000+k, las de la liga), los dos lados. P(serie) = probabilidad de ganar el bo3 con la regla del torneo (`league.series_win_prob`).

## D3a-distill

| rival | W-D-L oficiales | P(serie) oficiales | W-D-L aleatorios (8) | P(serie) aleatorios (8) |
|---|---|---|---|---|
| baseline:hard | 4-4-0 | 100 % | 14-2-0 | 100 % |
| A-g7_8000k | 0-8-0 | 48 % | 0-16-0 | 49 % |
| A-g6_4000k | 0-8-0 | 48 % | 0-16-0 | 49 % |
| A-g5_24000k | 6-2-0 | 100 % | 14-2-0 | 100 % |
| B-g4_3000k | 2-6-0 | 100 % | 4-12-0 | 100 % |
| C-g4_6000k | 2-6-0 | 100 % | 2-14-0 | 100 % |
| E-g1_2000k | 6-2-0 | 100 % | 12-4-0 | 100 % |
| R1 | 4-4-0 | 100 % | 4-12-0 | 100 % |
| A-g9_12006k | 0-8-0 | 48 % | 0-10-6 | 0 % |
| B-g7_6000k | 0-2-6 | 0 % | 4-8-4 | 48 % |
| C-g7_4500k | 0-8-0 | 48 % | 0-12-4 | 0 % |

## D2b-distill

| rival | W-D-L oficiales | P(serie) oficiales | W-D-L aleatorios (8) | P(serie) aleatorios (8) |
|---|---|---|---|---|
| baseline:hard | 6-2-0 | 100 % | 16-0-0 | 100 % |
| A-g7_8000k | 0-8-0 | 48 % | 0-14-2 | 0 % |
| A-g6_4000k | 0-8-0 | 48 % | 0-16-0 | 49 % |
| A-g5_24000k | 6-2-0 | 100 % | 12-4-0 | 100 % |
| B-g4_3000k | 0-8-0 | 48 % | 0-14-2 | 0 % |
| C-g4_6000k | 0-8-0 | 48 % | 2-14-0 | 100 % |
| E-g1_2000k | 8-0-0 | 100 % | 16-0-0 | 100 % |
| R1 | 4-4-0 | 100 % | 4-12-0 | 100 % |
| A-g9_12006k | 0-6-2 | 0 % | 0-8-8 | 0 % |
| B-g7_6000k | 0-0-8 | 0 % | 0-4-12 | 0 % |
| C-g7_4500k | 0-8-0 | 48 % | 0-12-4 | 0 % |
