def count_words(text: str) -> int:
    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    words = text.split()
    return len(words)
