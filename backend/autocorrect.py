"""
Spell-correction helper for the ASL typed-text output.

Kept separate from main.py so the correction logic can be tested and
modified independently of the WebSocket/inference pipeline.
"""

from spellchecker import SpellChecker

# Loaded once at import time — shared across all WebSocket connections.
_spell = SpellChecker()


def correct_last_word(typed_text: str) -> tuple[str, str | None, bool]:
    """
    Given the current typed_text (right before a space is appended),
    spell-checks the last finished word and corrects it if needed.

    Letters are signed/typed in uppercase, so correction is done
    case-insensitively and the result is written back in uppercase to
    match the rest of the typed text.

    Returns:
        (new_typed_text, finished_word, was_corrected)
        finished_word is the completed word (corrected or original), or
        None if there was no word to check (e.g. text was empty/just spaces).
        was_corrected indicates whether finished_word differs from the
        original signed word.
    """
    stripped = typed_text.rstrip(" ")
    trailing_spaces = typed_text[len(stripped):]
    words = stripped.split(" ")

    finished_word = None
    was_corrected = False
    if words and words[-1]:
        last_word = words[-1]
        finished_word = last_word
        correction = _spell.correction(last_word.lower())
        if correction and correction.lower() != last_word.lower():
            words[-1] = correction.upper()
            finished_word = correction.upper()
            was_corrected = True
            stripped = " ".join(words)

    new_typed_text = stripped + trailing_spaces + " "
    return new_typed_text, finished_word, was_corrected