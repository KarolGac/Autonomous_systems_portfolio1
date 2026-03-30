"""Hangman environment."""

import random
import math
import string
from dataclasses import dataclass, field
from game import HangmanGame


@dataclass
class StepResult:
    """Uitkomst van 1 actie in een speelronde."""
    observation: dict        # observatie na de actie
    reward: float            # beloning van deze stap
    terminated: bool         # True bij win of verlies
    info: dict = field(default_factory=dict)  # extra stapinformatie


class HangmanEnvironment:
    """
    Simpele single-agent environment voor galgje.

    Kern:
    - reset(...) start een nieuwe speelronde.
    - step(letter) verwerkt 1 lettergok.

    Beloningen (instelbaar):
    - correct_reward: beloning bij goede letter
    - win_reward: bonus bij winst
    - wrong_penalty: straf bij foute letter
    - lose_penalty: extra straf bij verlies
    - repeat_penalty: straf bij herhaalde gok

    Optionele reward shaping:
    - reveal_reward_scale: bonus per nieuw getoonde letter
    - info_gain_reward_scale: bonus als kandidaatset kleiner wordt
    - step_penalty: kleine straf per stap

    Stopvoorwaarde:
    - terminated=True: natuurlijke stop (gewonnen of verloren)
    """

    def __init__(
        self,
        word_list: list[str] | None = None,
        max_wrong_guesses: int = 10,
        correct_reward: float = 1.0,
        win_reward: float = 10.0,
        wrong_penalty: float = -1.0,
        lose_penalty: float = -10.0,
        repeat_penalty: float = -2.0,
        reveal_reward_scale: float = 0.0,
        info_gain_reward_scale: float = 0.0,
        step_penalty: float = 0.0,
        seed: int | None = None,
    ):
        self.word_list = word_list or []
        self.max_wrong_guesses = max_wrong_guesses

        # Instellingen voor beloning
        self.correct_reward = correct_reward
        self.win_reward = win_reward
        self.wrong_penalty = wrong_penalty
        self.lose_penalty = lose_penalty
        self.repeat_penalty = repeat_penalty
        self.reveal_reward_scale = reveal_reward_scale
        self.info_gain_reward_scale = info_gain_reward_scale
        self.step_penalty = step_penalty

        self._rng = random.Random(seed)
        self.game: HangmanGame | None = None
        self._current_word: str | None = None

    # Kerninterface

    def reset(
        self,
        word: str | None = None,
        *,
        seed: int | None = None,
    ) -> dict:
        """
        Start een nieuwe speelronde en geef de beginobservatie terug.

        Parameters
        ----------
        word : str, optional
            Geheime woord. Als None, kies random uit self.word_list.
        seed : int, optional
            Seed voor reproduceerbaarheid.
        Returns
        -------
        dict
            Observatie.
        """
        if seed is not None:
            self._rng.seed(seed)

        if word is not None:
            self._current_word = word.lower()
        elif self.word_list:
            self._current_word = self._rng.choice(self.word_list)
        else:
            raise ValueError(
                "Geen woord meegegeven en word_list is leeg. "
                "Geef een woord aan reset() of gebruik een word_list."
            )

        self.game = HangmanGame(self._current_word, self.max_wrong_guesses)
        return self.game.get_state()

    def step(self, letter: str) -> StepResult:
        """
        Verwerk 1 actie: raad een letter.

        Parameters
        ----------
        letter : str
            Een letter van a-z.

        Returns
        -------
        StepResult
            Nieuwe observatie, reward en statusvlaggen.
        """
        if self.game is None:
            raise RuntimeError("Roep reset() aan voor step().")

        letter = letter.lower()
        if len(letter) != 1 or letter not in string.ascii_lowercase:
            raise ValueError("Actie moet precies 1 letter van a-z zijn.")

        prev_state = self.game.get_state()
        prev_masked = prev_state["masked_word"]
        prev_candidates = self._candidate_count(prev_state)

        # Herhaalde gok: alleen repeat-penalty, status blijft verder gelijk.
        if letter in self.game.guessed_letters:
            obs = self.game.get_state()
            return StepResult(
                observation=obs,
                reward=self.repeat_penalty,
                terminated=self.game.is_over,
                info={"event": "repeat_guess", "letter": letter},
            )

        # Voer de gok uit in de game.
        correct = self.game.guess(letter)
        obs = self.game.get_state()

        # Basisbeloning op basis van de actie-uitkomst.
        if self.game.is_won:
            reward = self.correct_reward + self.win_reward
            event = "win"
        elif self.game.is_lost:
            reward = self.wrong_penalty + self.lose_penalty
            event = "loss"
        elif correct:
            reward = self.correct_reward
            event = "correct"
        else:
            reward = self.wrong_penalty
            event = "wrong"

        # Optionele shaping: extra bonus/straf bovenop de basisbeloning.
        shaping = {
            "reveal_bonus": 0.0,
            "info_gain_bonus": 0.0,
            "step_penalty": 0.0,
        }

        newly_revealed = sum(
            1 for prev, cur in zip(prev_masked, obs["masked_word"])
            if prev == "_" and cur != "_"
        )
        if newly_revealed > 0 and self.reveal_reward_scale > 0:
            shaping["reveal_bonus"] = self.reveal_reward_scale * newly_revealed

        if self.info_gain_reward_scale > 0 and prev_candidates > 0:
            next_candidates = self._candidate_count(obs)
            if next_candidates > 0 and next_candidates < prev_candidates:
                # Informatiewinst in bits: log2(vorig/nieuw).
                info_gain = math.log2(prev_candidates / next_candidates)
                shaping["info_gain_bonus"] = self.info_gain_reward_scale * info_gain
            else:
                next_candidates = max(next_candidates, 0)
            shaping["candidate_count_before"] = prev_candidates
            shaping["candidate_count_after"] = next_candidates

        if self.step_penalty != 0:
            shaping["step_penalty"] = -abs(self.step_penalty)

        reward += (
            shaping["reveal_bonus"]
            + shaping["info_gain_bonus"]
            + shaping["step_penalty"]
        )

        return StepResult(
            observation=obs,
            reward=reward,
            terminated=self.game.is_over,
            info={
                "event": event,
                "letter": letter,
                "correct": correct,
                "wrong_guesses": self.game.wrong_guesses,
                "newly_revealed": newly_revealed,
                "reward_components": shaping,
            },
        )

    def _candidate_count(self, state: dict) -> int:
        """
        Tel hoeveel woorden nog passen bij deze observatie.
        Geef 0 terug als word_list leeg is.
        """
        if not self.word_list:
            return 0

        masked = state["masked_word"]
        guessed = set(state["guessed_letters"])
        word_len = state["word_length"]

        count = 0
        for word in self.word_list:
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
                count += 1

        return count

    # Handige properties

    @property
    def is_done(self) -> bool:
        return self.game.is_over if self.game else False
