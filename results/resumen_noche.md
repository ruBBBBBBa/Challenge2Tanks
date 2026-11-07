# Resumen de la noche · PC1 (provisional, 19:40; se actualiza al terminar)

1. **Liga 11 y pausa:** liga 11 jugada (18:47–18:55) y pipeline parado antes de A-g12. Ningún A nuevo supera al candidato A-g9 en los tres criterios, pero **A-g11 (final)** es el más cercano al perfil buscado: 11 derrotas contra rivales de fuera y **hard 6-2-0 en oficiales** (`evaluate --games 4`: 86 %, hard 4-0-0).
2. **Dataset de D3 subido** (`cfdfcfe`): `results/d3play_replays.jsonl.gz`, 4,4 MB, con 10 752 partidas estocásticas, 8 129 decisivas y **2 505 064 pasos del ganador**, con la etiqueta determinista (LSTM incluida). Ningún modelo pasa del 35 % (máximo A-g9, 27 %); sin R1. PC2 lo reconstruye con `winners_dataset.py --from-play`.
3. **Pipeline relanzado a las 19:24** (PID 31353, `logs/pipeline5.out`): liga 12 → **A-g12 desde A-g11_8000k** (Elo 1818). 8 generaciones de 12M (A-g12 a A-g19), PFSP hard/variance alternos; la última debería terminar hacia las 7:50. El sync (PID 6564) publica cada A elegido para los explotadores de PC3.
4. **Jueces:** el juez automático ya usa R1 y **R2-g5_3000k** (subido por PC4 a `models/reserved/`; nunca en ningún pool). En la liga 12, los tres mejores A ganan el 100 % de las series contra los dos.
5. **Comparación con el candidato en cada liga:** se va añadiendo a `results/comparaciones_noche.md`. La entrega provisional sigue siendo A-g9 hasta que decidamos.
