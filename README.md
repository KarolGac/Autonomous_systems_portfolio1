# Hangman — Rule-Based Agent Vergelijking

Een stapsgewijze vergelijking van vijf rule-based AI-strategieën voor het spel Hangman, van puur willekeurig tot informatietheorie-gebaseerd.

## Projectstructuur

```
game_theory_project/
├── game.py           # Kernlogica van het spel (HangmanGame)
├── environment.py    # RL-omgeving met reward-systeem (HangmanEnvironment)
├── agents.py         # Vijf agent-strategieën
├── benchmark.py      # Benchmark-runner en evaluatiefuncties
├── main.ipynb        # Notebook met uitleg, demo's en visualisaties
├── words.txt         # Woordenlijst (4–12 letters)
└── requirements.txt  # Python-afhankelijkheden
```

## De vijf strategieën

| # | Agent | Kernidee |
|---|---|---|
| 1 | **Random** | Kiest willekeurig een letter |
| 2 | **Frequency** | Volgt Engelse letterfrequentie (e, t, a, o, …) |
| 3 | **Positional Frequency** | Letterfrequentie per positie in de woordenlijst |
| 4 | **Word List Elimination** | Filtert onmogelijke woorden na elke gok |
| 5 | **Entropy** | Kiest de letter met de hoogste Shannon-entropie |

## Installatie

```bash
pip install -r requirements.txt
```

## Gebruik

Open `main.ipynb` in Jupyter en voer de cellen van boven naar beneden uit:

```bash
jupyter notebook main.ipynb
```

Of gebruik de benchmark direct via de terminal:

```bash
python benchmark.py
```

## Resultaten (samenvatting)

- **Kleine woordenlijst** (< 5.000 woorden): Word List Elimination scoort het best
- **Grote woordenlijst** (> 10.000 woorden): Entropy Agent wint door informatieoptimale splitsing
- Het centrale inzicht: de agent die zijn onzekerheid het snelst reduceert, wint het vaakst

## Vereisten

- Python 3.10+
- Zie `requirements.txt` voor pakketversies
