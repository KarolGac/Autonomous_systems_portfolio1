"""Galgje-agents."""

import random
import math
from abc import ABC, abstractmethod
from collections import Counter


class BaseAgent(ABC):
    """Basisklasse voor alle agents."""

    name: str = "BaseAgent"

    @abstractmethod
    def pick_letter(self, state: dict, word_list: list[str] | None = None) -> str:
        """Geef op basis van de spelstatus de volgende letter terug."""
        ...


# ---------------------------------------------------------------------------
# Willekeurig
# ---------------------------------------------------------------------------
class RandomAgent(BaseAgent):
    """Kiest een willekeurige letter uit de overgebleven opties."""

    name = "Random Agent"

    def pick_letter(self, state: dict, word_list=None) -> str:
        return random.choice(state["remaining_letters"])


# ---------------------------------------------------------------------------
# Frequentie
# ---------------------------------------------------------------------------
class FrequencyAgent(BaseAgent):
    """Raadt letters op basis van Engelse letterfrequentie."""

    name = "Frequency Agent"

    # Engelse volgorde van letterfrequentie (hoog naar laag)
    FREQ_ORDER = list("etaoinshrdlcumwfgypbvkjxqz")

    def pick_letter(self, state: dict, word_list=None) -> str:
        remaining = set(state["remaining_letters"])
        for letter in self.FREQ_ORDER:
            if letter in remaining:
                return letter
        return state["remaining_letters"][0]


# ---------------------------------------------------------------------------
# Positie-frequentie
# ---------------------------------------------------------------------------
class PositionalFrequencyAgent(BaseAgent):
    """
    Geeft letters punten op basis van hoe vaak ze staan op onbekende
    posities in woorden met dezelfde lengte.
    """

    name = "Positional Frequency Agent"

    def pick_letter(self, state: dict, word_list=None) -> str:
        remaining = set(state["remaining_letters"])
        masked = state["masked_word"]
        word_len = state["word_length"]

        if not word_list:
            # Terugval naar simpele frequentie
            return FrequencyAgent().pick_letter(state)

        # Filter woorden op gelijke lengte
        candidates = [w for w in word_list if len(w) == word_len]
        if not candidates:
            return FrequencyAgent().pick_letter(state)

        # Zoek onbekende posities
        unknown_positions = [i for i, ch in enumerate(masked) if ch == "_"]

        # Geef letters punten op onbekende posities
        scores: Counter = Counter()
        for word in candidates:
            for pos in unknown_positions:
                ch = word[pos]
                if ch in remaining:
                    scores[ch] += 1

        if scores:
            return scores.most_common(1)[0][0]
        return FrequencyAgent().pick_letter(state)


# ---------------------------------------------------------------------------
# Woordenlijst-eliminatie
# ---------------------------------------------------------------------------
class WordListEliminationAgent(BaseAgent):
    """
    Houdt een lijst met mogelijke woorden bij. Na elke gok vallen
    onmogelijke woorden af. Kiest de letter in de meeste overgebleven woorden.
    """

    name = "Word List Elimination Agent"

    def _filter_candidates(self, candidates: list[str], state: dict) -> list[str]:
        """Filter op woorden die passen bij de huidige spelstatus."""
        masked = state["masked_word"]
        guessed = set(state["guessed_letters"])
        word_len = state["word_length"]

        filtered = []
        for word in candidates:
            if len(word) != word_len:
                continue

            match = True
            for i, ch in enumerate(masked):
                if ch != "_":
                    # Bekende positie: letter moet hier passen
                    if word[i] != ch:
                        match = False
                        break
                else:
                    # Onbekende positie: hier mag geen al gegokte letter staan
                    if word[i] in guessed:
                        match = False
                        break

            if match:
                filtered.append(word)

        return filtered

    def pick_letter(self, state: dict, word_list=None) -> str:
        remaining = set(state["remaining_letters"])

        if not word_list:
            return FrequencyAgent().pick_letter(state)

        candidates = self._filter_candidates(word_list, state)

        if not candidates:
            return FrequencyAgent().pick_letter(state)

        # Kies letter die in de meeste kandidaatwoorden voorkomt
        letter_word_count: Counter = Counter()
        for word in candidates:
            unique_letters = set(word) & remaining
            for ch in unique_letters:
                letter_word_count[ch] += 1

        if letter_word_count:
            return letter_word_count.most_common(1)[0][0]
        return FrequencyAgent().pick_letter(state)


# ---------------------------------------------------------------------------
# Entropie / informatiewinst
# ---------------------------------------------------------------------------
class EntropyAgent(BaseAgent):
    """
    Kiest de letter met de meeste informatiewinst (Shannon-entropie).
    Verdeelt kandidaten op basis van het patroon dat een letter zou tonen
    en kiest de letter met de hoogste entropie.
    Bij het laatste leven weegt overleven zwaarder mee.
    """

    name = "Entropy Agent"

    def _filter_candidates(self, candidates: list[str], state: dict) -> list[str]:
        """Zelfde filtering als WordListEliminationAgent."""
        masked = state["masked_word"]
        guessed = set(state["guessed_letters"])
        word_len = state["word_length"]

        filtered = []
        for word in candidates:
            if len(word) != word_len:
                continue
            match = True
            for i, ch in enumerate(masked):
                if ch != "_":
                    if word[i] != ch:
                        match = False
                        break
                else:
                    if word[i] in guessed:
                        match = False
                        break
            if match:
                filtered.append(word)
        return filtered

    def _get_pattern(self, word: str, letter: str, unknown_positions: list[int]) -> tuple:
        """
        Geef een tuple terug met waar de letter op onbekende posities staat.
        Alleen nog verborgen posities tellen mee.
        """
        return tuple(word[i] == letter for i in unknown_positions)

    def pick_letter(self, state: dict, word_list=None) -> str:
        remaining = set(state["remaining_letters"])

        if not word_list:
            return FrequencyAgent().pick_letter(state)

        candidates = self._filter_candidates(word_list, state)

        if not candidates:
            return FrequencyAgent().pick_letter(state)

        masked = state["masked_word"]
        unknown_positions = [i for i, ch in enumerate(masked) if ch == "_"]

        n = len(candidates)
        lives_left = state["max_wrong_guesses"] - state["wrong_guesses"]

        # Bekijk alleen letters die echt in kandidaten voorkomen
        candidate_letters = set()
        for word in candidates:
            for i in unknown_positions:
                if word[i] in remaining:
                    candidate_letters.add(word[i])

        letters_to_check = candidate_letters if candidate_letters else remaining
        miss_pattern = tuple(False for _ in unknown_positions)

        best_letter = None
        best_score = -float('inf')

        for letter in letters_to_check:
            # Verdeel kandidaten op basis van het patroon van deze letter
            pattern_counts: Counter = Counter()
            for word in candidates:
                pattern = self._get_pattern(word, letter, unknown_positions)
                pattern_counts[pattern] += 1

            # Shannon-entropie van deze verdeling
            entropy = 0.0
            for count in pattern_counts.values():
                if count > 0:
                    p = count / n
                    entropy -= p * math.log2(p)

            p_miss = pattern_counts.get(miss_pattern, 0) / n

            if lives_left == 1:
                # Laatste leven: een misser is direct game over.
                # Score = kans op overleven x (1 + info bij hits)
                if p_miss >= 1.0:
                    score = -1.0
                else:
                    p_hit = 1.0 - p_miss
                    n_hit = n - pattern_counts.get(miss_pattern, 0)
                    hit_entropy = 0.0
                    for pat, cnt in pattern_counts.items():
                        if pat != miss_pattern:
                            p_c = cnt / n_hit
                            hit_entropy -= p_c * math.log2(p_c)
                    score = p_hit * (1.0 + hit_entropy)
            else:
                        # Genoeg levens: maximale entropie.
                        # Kleine hit-bonus breekt gelijke scores.
                score = entropy + 1e-4 * (1.0 - p_miss)

            if score > best_score:
                best_score = score
                best_letter = letter

        return best_letter if best_letter else FrequencyAgent().pick_letter(state)


# ---------------------------------------------------------------------------
# Helper: geef alle agents
# ---------------------------------------------------------------------------
def get_all_agents() -> list[BaseAgent]:
    return [
        RandomAgent(),
        FrequencyAgent(),
        PositionalFrequencyAgent(),
        WordListEliminationAgent(),
        EntropyAgent(),
    ]
