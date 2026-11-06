# Evaluación determinista (40 partidas por rival: 16 official_aug + 4 aleatorios × 2 lados)

W-D-L por rival. Referencia: `models/shared/A-g4_4000k.zip`.

| modelo | A-g4_4000k | B-g1_20000k | C-g1_20000k | baseline:hard | total W-D-L |
|---|---|---|---|---|---|
| `E-g1_10000k.zip` | 40-0-0 | 34-6-0 | 34-6-0 | 40-0-0 | 148-12-0 |
| `E-g1_12000k.zip` | 36-0-4 | 40-0-0 | 34-6-0 | 36-4-0 | 146-10-4 |
| `E-g1_2000k.zip` **← mejor** | 36-4-0 | 38-2-0 | 38-2-0 | 40-0-0 | 152-8-0 |
| `E-g1_4000k.zip` | 32-8-0 | 36-4-0 | 32-8-0 | 38-2-0 | 138-22-0 |
| `E-g1_6000k.zip` | 36-2-2 | 40-0-0 | 34-6-0 | 40-0-0 | 150-8-2 |
| `E-g1_8000k.zip` | 38-2-0 | 40-0-0 | 32-8-0 | 40-0-0 | 150-10-0 |
| `A-g4_4000k.zip` (ref) | 0-40-0 | 36-4-0 | 24-16-0 | 38-2-0 | 98-62-0 |

Mejor (menos empates sin perder más que la referencia, 0 derrotas): `models/league/E-g1_2000k.zip`
