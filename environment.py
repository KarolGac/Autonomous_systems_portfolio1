"""Hangman Environment — RL-style wrapper around HangmanGame."""

import random
from dataclasses import dataclass, field
from game import HangmanGame


@dataclass
class StepResult:
    """Returned by HangmanEnvironment.step()."""
    observation: dict        # full game state dict
    reward: float            # reward for this step
    done: bool               # True when episode is over
    info: dict = field(default_factory=dict)  # extra diagnostics


class HangmanEnvironment:
    """
    RL-style environment for Hangman.

    Reward structure (all configurable via constructor):
      +correct_reward   — for each correct letter guess
      +win_reward       — bonus on winning
      -wrong_penalty    — for each wrong guess
      -lose_penalty     — penalty on losing
      -repeat_penalty   — for guessing an already-guessed letter
                          (should not happen when using state['remaining_letters'])

    Typical usage
    -------------
        env = HangmanEnvironment(word_list=words)
        obs = env.reset()
        done = False
        while not done:
            letter = agent.pick_letter(obs, env.word_list)
            result = env.step(letter)
            obs, reward, done = result.observation, result.reward, result.done
        print("Won!" if env.game.is_won else "Lost.")
    """

    def __init__(
        self,
        word_list: list[str] | None = None,
        max_wrong_guesses: int = 8,
        correct_reward: float = 1.0,
        win_reward: float = 10.0,
        wrong_penalty: float = -1.0,
        lose_penalty: float = -10.0,
        repeat_penalty: float = -2.0,
        seed: int | None = None,
    ):
        self.word_list = word_list or []
        self.max_wrong_guesses = max_wrong_guesses

        # Reward config
        self.correct_reward = correct_reward
        self.win_reward = win_reward
        self.wrong_penalty = wrong_penalty
        self.lose_penalty = lose_penalty
        self.repeat_penalty = repeat_penalty

        self._rng = random.Random(seed)
        self.game: HangmanGame | None = None
        self._current_word: str | None = None

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self, word: str | None = None) -> dict:
        """
        Start a new episode.

        Parameters
        ----------
        word : str, optional
            Secret word to use. If None, a random word is sampled from
            ``self.word_list``. Raises ValueError if no word can be chosen.

        Returns
        -------
        dict
            Initial observation (game state).
        """
        if word is not None:
            self._current_word = word.lower()
        elif self.word_list:
            self._current_word = self._rng.choice(self.word_list)
        else:
            raise ValueError(
                "No word provided and word_list is empty. "
                "Pass a word to reset() or supply a word_list."
            )

        self.game = HangmanGame(self._current_word, self.max_wrong_guesses)
        return self.game.get_state()

    def step(self, letter: str) -> StepResult:
        """
        Apply one action (guess a letter).

        Parameters
        ----------
        letter : str
            A single lowercase letter.

        Returns
        -------
        StepResult
            observation, reward, done, info
        """
        if self.game is None:
            raise RuntimeError("Call reset() before step().")

        letter = letter.lower()

        # --- handle repeat guess ---
        if letter in self.game.guessed_letters:
            obs = self.game.get_state()
            return StepResult(
                observation=obs,
                reward=self.repeat_penalty,
                done=self.game.is_over,
                info={"event": "repeat_guess", "letter": letter},
            )

        # --- apply guess ---
        correct = self.game.guess(letter)
        obs = self.game.get_state()

        # --- calculate reward ---
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

        return StepResult(
            observation=obs,
            reward=reward,
            done=self.game.is_over,
            info={
                "event": event,
                "letter": letter,
                "correct": correct,
                "wrong_guesses": self.game.wrong_guesses,
            },
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> str:
        """Return a human-readable string of the current game state."""
        if self.game is None:
            return "[No active game — call reset() first]"

        g = self.game
        wrong_left = g.max_wrong_guesses - g.wrong_guesses
        wrong_letters = sorted(
            ch for ch in g.guessed_letters if ch not in g.secret_word
        )
        correct_letters = sorted(
            ch for ch in g.guessed_letters if ch in g.secret_word
        )

        lines = [
            _hangman_art(g.wrong_guesses),
            f"  Word   : {g.masked_word}  ({len(g.secret_word)} letters)",
            f"  Wrong  : {', '.join(wrong_letters) or '—'}  "
            f"({g.wrong_guesses}/{g.max_wrong_guesses} — {wrong_left} left)",
            f"  Correct: {', '.join(correct_letters) or '—'}",
        ]

        if g.is_won:
            lines.append(f"\n  *** YOU WIN! The word was '{g.secret_word}' ***")
        elif g.is_lost:
            lines.append(f"\n  *** GAME OVER. The word was '{g.secret_word}' ***")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def word_length(self) -> int | None:
        return len(self._current_word) if self._current_word else None

    @property
    def is_done(self) -> bool:
        return self.game.is_over if self.game else False


# ---------------------------------------------------------------------------
# ASCII hangman art helper
# ---------------------------------------------------------------------------
_HANGMAN_STAGES = [
    # 0 wrong
    ("  +---+  ",
     "  |   |  ",
     "      |  ",
     "      |  ",
     "      |  ",
     "      |  ",
     "========="),
    # 1 wrong — head
    ("  +---+  ",
     "  |   |  ",
     "  O   |  ",
     "      |  ",
     "      |  ",
     "      |  ",
     "========="),
    # 2 wrong — body
    ("  +---+  ",
     "  |   |  ",
     "  O   |  ",
     "  |   |  ",
     "      |  ",
     "      |  ",
     "========="),
    # 3 wrong — left arm
    ("  +---+  ",
     "  |   |  ",
     "  O   |  ",
     " /|   |  ",
     "      |  ",
     "      |  ",
     "========="),
    # 4 wrong — both arms
    ("  +---+  ",
     "  |   |  ",
     "  O   |  ",
     " /|\\  |  ",
     "      |  ",
     "      |  ",
     "========="),
    # 5 wrong — left leg
    ("  +---+  ",
     "  |   |  ",
     "  O   |  ",
     " /|\\  |  ",
     " /    |  ",
     "      |  ",
     "========="),
    # 6 wrong — both legs (dead)
    ("  +---+  ",
     "  |   |  ",
     "  O   |  ",
     " /|\\  |  ",
     " / \\  |  ",
     "      |  ",
     "========="),
]


def _hangman_art(wrong_guesses: int) -> str:
    stage = min(wrong_guesses, len(_HANGMAN_STAGES) - 1)
    return "\n".join(_HANGMAN_STAGES[stage])
