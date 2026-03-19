"""
Hangman Agent Benchmark
=======================
Runs all agents against a set of test words and compares performance.

Metrics:
  - Win rate (%)
  - Average wrong guesses per game
  - Average total guesses per game
"""

import random
import time
from collections import defaultdict

from game import HangmanGame
from agents import get_all_agents, BaseAgent


# ---------------------------------------------------------------------------
# Word list loader
# ---------------------------------------------------------------------------
def load_word_list(path: str = "/usr/share/dict/american-english") -> list[str]:
    """Load a word list, filtering to lowercase alpha words of length 4-12."""
    words = []
    try:
        with open(path, "r") as f:
            for line in f:
                word = line.strip().lower()
                if word.isalpha() and 4 <= len(word) <= 12:
                    words.append(word)
    except FileNotFoundError:
        # Fallback: curated 200-word list for strategy evaluation
        words = [
            # programming / CS
            "python", "hangman", "strategy", "algorithm", "entropy",
            "baseline", "frequency", "position", "random", "agent",
            "computer", "science", "machine", "learning", "model",
            "neural", "network", "tensor", "matrix", "vector",
            "kernel", "feature", "sample", "dataset", "training",
            "abstract", "binary", "compile", "debug", "execute",
            "function", "global", "integer", "library", "module",
            "object", "pointer", "runtime", "syntax", "variable",
            "boolean", "cluster", "cipher", "commit", "branch",
            "deploy", "docker", "server", "socket", "thread",
            "process", "buffer", "cache", "queue", "stack",
            "graph", "search", "index", "parse", "token",
            "lambda", "yield", "async", "class", "method",
            "import", "export", "struct", "union", "bitwise",
            "encode", "decode", "signal", "filter", "layer",
            # nature / science
            "forest", "jungle", "desert", "arctic", "tundra",
            "glacier", "volcano", "canyon", "plateau", "valley",
            "ocean", "river", "marsh", "delta", "spring",
            "comet", "planet", "galaxy", "nebula", "quasar",
            "proton", "neutron", "photon", "electron", "plasma",
            "carbon", "oxygen", "nitrogen", "hydrogen", "helium",
            "crystal", "mineral", "fossil", "amber", "quartz",
            "typhoon", "monsoon", "tornado", "blizzard", "drought",
            # animals
            "penguin", "dolphin", "panther", "leopard", "cheetah",
            "gorilla", "buffalo", "flamingo", "peacock", "vulture",
            "lobster", "octopus", "jellyfish", "hamster", "rabbit",
            "sparrow", "falcon", "eagle", "parrot", "toucan",
            "piranha", "salmon", "marlin", "shrimp", "oyster",
            # everyday / objects
            "blanket", "candle", "mirror", "carpet", "pillow",
            "cabinet", "drawer", "lantern", "compass", "ladder",
            "suitcase", "umbrella", "backpack", "wallet", "helmet",
            "bicycle", "engine", "rocket", "capsule", "antenna",
            "battery", "circuit", "magnet", "turbine", "piston",
            # food / drinks
            "mango", "papaya", "walnut", "almond", "cashew",
            "noodle", "spaghetti", "burrito", "pancake", "waffle",
            "brownie", "custard", "popcorn", "pretzel", "cracker",
            "mustard", "ketchup", "vinegar", "ginger", "pepper",
            "espresso", "cappuccino", "lemonade", "smoothie", "cocktail",
            # places / travel
            "airport", "harbour", "station", "bridge", "tunnel",
            "castle", "palace", "pyramid", "temple", "cathedral",
            "village", "suburb", "island", "peninsula", "fjord",
            "highway", "avenue", "square", "market", "stadium",
            # abstract / language
            "freedom", "justice", "balance", "harmony", "wisdom",
            "courage", "loyalty", "mystery", "fantasy", "legend",
            "chapter", "phrase", "grammar", "dialect", "accent",
            "synonym", "analogy", "paradox", "riddle", "fable",
            # health / body
            "muscle", "tissue", "neuron", "artery", "tendon",
            "vitamin", "protein", "insulin", "hormone", "plasma",
            # extra variety
            "lantern", "trumpet", "rhythm", "puzzle", "trophy",
            "journal", "sketch", "marble", "velvet", "amber",
            "crimson", "magenta", "silver", "golden", "cobalt",
        ]
    # Deduplicate
    return list(set(words))


# ---------------------------------------------------------------------------
# Run a single game
# ---------------------------------------------------------------------------
def run_game(agent: BaseAgent, secret_word: str, word_list: list[str],
             max_wrong: int = 10, verbose: bool = False) -> dict:
    """Play one game of Hangman with the given agent. Return result dict."""
    game = HangmanGame(secret_word, max_wrong_guesses=max_wrong)

    while not game.is_over:
        state = game.get_state()
        letter = agent.pick_letter(state, word_list)

        if verbose:
            print(f"  {game.masked_word}  | guess: '{letter}'", end="")

        correct = game.guess(letter)

        if verbose:
            print(f"  -> {'✓' if correct else '✗'}  (wrong: {game.wrong_guesses}/{max_wrong})")

    result = {
        "word": secret_word,
        "won": game.is_won,
        "wrong_guesses": game.wrong_guesses,
        "total_guesses": len(game.guessed_letters),
    }

    if verbose:
        status = "WON ✓" if game.is_won else "LOST ✗"
        print(f"  >> {status}  word: {secret_word}  "
              f"(wrong: {game.wrong_guesses}, total: {len(game.guessed_letters)})\n")

    return result


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def benchmark(n_games: int = 200, max_wrong: int = 6, seed: int = 42,
              verbose_games: int = 3):
    """Run all agents on the same set of random words and print results."""
    random.seed(seed)

    word_list = load_word_list()
    print(f"Loaded {len(word_list)} words in word list.\n")

    # Sample test words
    test_words = random.sample(word_list, min(n_games, len(word_list)))
    agents = get_all_agents()

    print("=" * 72)
    print(f"  HANGMAN AGENT BENCHMARK  —  {len(test_words)} games, max {max_wrong} wrong guesses")
    print("=" * 72)

    all_results: dict[str, list[dict]] = {}

    for agent in agents:
        print(f"\n{'─' * 72}")
        print(f"  Agent: {agent.name}")
        print(f"{'─' * 72}")

        results = []
        start = time.time()

        for i, word in enumerate(test_words):
            verbose = i < verbose_games
            if verbose:
                print(f"\n  Game {i + 1}: '{word}' ({len(word)} letters)")
            result = run_game(agent, word, word_list, max_wrong, verbose=verbose)
            results.append(result)

        elapsed = time.time() - start
        all_results[agent.name] = results

        # Stats
        wins = sum(r["won"] for r in results)
        win_rate = wins / len(results) * 100
        avg_wrong = sum(r["wrong_guesses"] for r in results) / len(results)
        avg_total = sum(r["total_guesses"] for r in results) / len(results)

        print(f"\n  ... ({len(test_words) - verbose_games} more games played)")
        print(f"\n  Results: {wins}/{len(results)} won ({win_rate:.1f}%)")
        print(f"  Avg wrong guesses: {avg_wrong:.2f}")
        print(f"  Avg total guesses: {avg_total:.2f}")
        print(f"  Time: {elapsed:.2f}s")

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print(f"\n\n{'=' * 72}")
    print("  SUMMARY")
    print(f"{'=' * 72}")
    print(f"\n  {'Agent':<30} {'Win Rate':>10} {'Avg Wrong':>12} {'Avg Total':>12}")
    print(f"  {'─' * 30} {'─' * 10} {'─' * 12} {'─' * 12}")

    # Sort by win rate descending
    ranking = []
    for name, results in all_results.items():
        wins = sum(r["won"] for r in results)
        win_rate = wins / len(results) * 100
        avg_wrong = sum(r["wrong_guesses"] for r in results) / len(results)
        avg_total = sum(r["total_guesses"] for r in results) / len(results)
        ranking.append((name, win_rate, avg_wrong, avg_total))

    ranking.sort(key=lambda x: (-x[1], x[2]))

    for i, (name, wr, aw, at) in enumerate(ranking):
        medal = ["🥇", "🥈", "🥉", "  ", "  "][i]
        print(f"  {medal} {name:<28} {wr:>9.1f}% {aw:>11.2f} {at:>11.2f}")

    print(f"\n{'=' * 72}")
    print("  Benchmark complete!")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    benchmark(n_games=200, max_wrong=6, seed=42, verbose_games=2)
