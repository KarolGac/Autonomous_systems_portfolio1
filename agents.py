"""Hangman Agents: 1 baseline + 4 rule-based strategies."""

import random
import math
from abc import ABC, abstractmethod
from collections import Counter


class BaseAgent(ABC):
    """Abstract base class for all Hangman agents."""

    name: str = "BaseAgent"

    @abstractmethod
    def pick_letter(self, state: dict, word_list: list[str] | None = None) -> str:
        """Given the current game state, return the next letter to guess."""
        ...


# ---------------------------------------------------------------------------
# Baseline: Random Agent
# ---------------------------------------------------------------------------
class RandomAgent(BaseAgent):
    """Picks a random letter from the remaining options. Pure baseline."""

    name = "Random Agent"

    def pick_letter(self, state: dict, word_list=None) -> str:
        return random.choice(state["remaining_letters"])


# ---------------------------------------------------------------------------
# Agent 1: Frequency Agent
# ---------------------------------------------------------------------------
class FrequencyAgent(BaseAgent):
    """Guesses letters in order of English language frequency."""

    name = "Frequency Agent"

    # English letter frequency order (most to least common)
    FREQ_ORDER = list("etaoinshrdlcumwfgypbvkjxqz")

    def pick_letter(self, state: dict, word_list=None) -> str:
        remaining = set(state["remaining_letters"])
        for letter in self.FREQ_ORDER:
            if letter in remaining:
                return letter
        return state["remaining_letters"][0]


# ---------------------------------------------------------------------------
# Agent 2: Positional Frequency Agent
# ---------------------------------------------------------------------------
class PositionalFrequencyAgent(BaseAgent):
    """
    Scores each letter by how often it appears at each UNKNOWN position
    among words of the same length in the word list.
    """

    name = "Positional Frequency Agent"

    def pick_letter(self, state: dict, word_list=None) -> str:
        remaining = set(state["remaining_letters"])
        masked = state["masked_word"]
        word_len = state["word_length"]

        if not word_list:
            # Fallback to simple frequency
            return FrequencyAgent().pick_letter(state)

        # Filter words matching length
        candidates = [w for w in word_list if len(w) == word_len]
        if not candidates:
            return FrequencyAgent().pick_letter(state)

        # Find unknown positions
        unknown_positions = [i for i, ch in enumerate(masked) if ch == "_"]

        # Score letters by positional frequency at unknown positions
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
# Agent 3: Word List Elimination Agent
# ---------------------------------------------------------------------------
class WordListEliminationAgent(BaseAgent):
    """
    Maintains a list of possible words. After each guess, eliminates
    impossible words. Picks the letter occurring in the most remaining words.
    """

    name = "Word List Elimination Agent"

    def _filter_candidates(self, candidates: list[str], state: dict) -> list[str]:
        """Filter word list to only words matching the current game state."""
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
                    # Known position: word must have this letter here
                    if word[i] != ch:
                        match = False
                        break
                else:
                    # Unknown position: word must NOT have a guessed letter here
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

        # Pick the letter that appears in the most candidate words
        letter_word_count: Counter = Counter()
        for word in candidates:
            unique_letters = set(word) & remaining
            for ch in unique_letters:
                letter_word_count[ch] += 1

        if letter_word_count:
            return letter_word_count.most_common(1)[0][0]
        return FrequencyAgent().pick_letter(state)


# ---------------------------------------------------------------------------
# Agent 4: Entropy / Information Gain Agent
# ---------------------------------------------------------------------------
class EntropyAgent(BaseAgent):
    """
    Picks the letter that maximizes information gain (entropy reduction).
    For each candidate letter, calculates the entropy of the resulting
    partition of remaining words.
    """

    name = "Entropy Agent"

    def _filter_candidates(self, candidates: list[str], state: dict) -> list[str]:
        """Same filtering as WordListEliminationAgent."""
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
        Return a tuple showing where 'letter' appears at unknown positions.
        Only considers positions that are still hidden — this creates
        a more meaningful partition of remaining words.
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
        best_letter = None
        best_info_gain = -1.0

        # Only consider letters that actually appear in candidates
        candidate_letters = set()
        for word in candidates:
            for i in unknown_positions:
                if word[i] in remaining:
                    candidate_letters.add(word[i])

        # Also consider letters NOT in any candidate (they'd eliminate all at once — 0 entropy)
        # We only want letters that create a meaningful split
        letters_to_check = candidate_letters if candidate_letters else remaining

        for letter in letters_to_check:
            # Partition candidates by the pattern this letter would reveal
            pattern_counts: Counter = Counter()
            for word in candidates:
                pattern = self._get_pattern(word, letter, unknown_positions)
                pattern_counts[pattern] += 1

            # Calculate entropy of the partition
            entropy = 0.0
            for count in pattern_counts.values():
                if count > 0:
                    p = count / n
                    entropy -= p * math.log2(p)

            if entropy > best_info_gain:
                best_info_gain = entropy
                best_letter = letter

        return best_letter if best_letter else FrequencyAgent().pick_letter(state)


# ---------------------------------------------------------------------------
# Helper: get all agents
# ---------------------------------------------------------------------------
def get_all_agents() -> list[BaseAgent]:
    return [
        RandomAgent(),
        FrequencyAgent(),
        PositionalFrequencyAgent(),
        WordListEliminationAgent(),
        EntropyAgent(),
    ]
