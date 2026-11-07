# Evaluación de los snapshots de D2-g1

Determinista, con `student/eval_vs.py`: 4 mapas oficiales y 8 aleatorios de la liga, los dos lados. Cada celda es «oficiales · aleatorios» en W-D-L. R1 solo como rival de evaluación.

| snapshot | baseline:hard | A-g7_8000k | A-g6_4000k | A-g5_24000k | B-g4_3000k | C-g4_6000k | E-g1_2000k | R1 | total oficiales | total aleatorios |
|---|---|---|---|---|---|---|---|---|---|---|
| D2b-distill | 6-2-0 · 16-0-0 | 0-8-0 · 0-14-2 | 0-8-0 · 0-16-0 | 6-2-0 · 12-4-0 | 0-8-0 · 0-14-2 | 0-8-0 · 2-14-0 | 8-0-0 · 16-0-0 | 4-4-0 · 4-12-0 | 24-40-0 | 50-74-4 |
