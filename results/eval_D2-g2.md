# Evaluación de los snapshots de D2-g2

Determinista, con `student/eval_vs.py`: 4 mapas oficiales y 8 aleatorios de la liga, los dos lados. Cada celda es «oficiales · aleatorios» en W-D-L. R1 solo como rival de evaluación.

| snapshot | baseline:hard | A-g17_8000k | A-g15_12006k | B-g7_6000k | C-g7_4500k | R1 | total oficiales | total aleatorios |
|---|---|---|---|---|---|---|---|---|
| D4-distill | 4-4-0 · 12-4-0 | 0-8-0 · 2-14-0 | 0-8-0 · 0-14-2 | 2-4-2 · 12-4-0 | 0-6-2 · 2-12-2 | 4-4-0 · 16-0-0 | 10-34-4 | 44-48-4 |
| D2-g2_2000k | 0-8-0 · 4-12-0 | 0-8-0 · 0-14-2 | 0-6-2 · 0-14-2 | 0-8-0 · 2-14-0 | 0-8-0 · 0-12-4 | 8-0-0 · 2-14-0 | 8-38-2 | 8-80-8 |
| D2-g2_4000k | 4-4-0 · 8-8-0 | 0-8-0 · 0-16-0 | 0-8-0 · 0-16-0 | 0-6-2 · 4-6-6 | 0-8-0 · 0-14-2 | 8-0-0 · 12-4-0 | 12-34-2 | 24-64-8 |
