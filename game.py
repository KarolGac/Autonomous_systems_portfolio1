"""Hangman Game Engine."""

class HangmanGame:
    """Core Hangman game logic."""

    def __init__(self, secret_word: str, max_wrong_guesses: int = 8):
        self.secret_word = secret_word.lower()
        self.max_wrong_guesses = max_wrong_guesses
        self.guessed_letters: set[str] = set()
        self.wrong_guesses: int = 0

    @property
    def masked_word(self) -> str:
        """Return the word with unguessed letters as underscores."""
        return "".join(
            ch if ch in self.guessed_letters else "_"
            for ch in self.secret_word
        )

    @property
    def remaining_letters(self) -> list[str]:
        """Letters that haven't been guessed yet."""
        return sorted(set("abcdefghijklmnopqrstuvwxyz") - self.guessed_letters)

    @property
    def is_won(self) -> bool:
        return all(ch in self.guessed_letters for ch in self.secret_word)

    @property
    def is_lost(self) -> bool:
        return self.wrong_guesses >= self.max_wrong_guesses

    @property
    def is_over(self) -> bool:
        return self.is_won or self.is_lost

    def guess(self, letter: str) -> bool:
        """
        Guess a letter. Returns True if the letter is in the word.
        Raises ValueError if letter was already guessed or game is over.
        """
        letter = letter.lower()
        if self.is_over:
            raise ValueError("Game is already over.")
        if letter in self.guessed_letters:
            raise ValueError(f"Letter '{letter}' was already guessed.")
        if letter not in "abcdefghijklmnopqrstuvwxyz":
            raise ValueError(f"Invalid letter: '{letter}'")

        self.guessed_letters.add(letter)
        if letter in self.secret_word:
            return True
        else:
            self.wrong_guesses += 1
            return False

    def get_state(self) -> dict:
        """Return a snapshot of the current game state."""
        return {
            "masked_word": self.masked_word,
            "word_length": len(self.secret_word),
            "guessed_letters": sorted(self.guessed_letters),
            "remaining_letters": self.remaining_letters,
            "wrong_guesses": self.wrong_guesses,
            "max_wrong_guesses": self.max_wrong_guesses,
            "is_won": self.is_won,
            "is_lost": self.is_lost,
        }
