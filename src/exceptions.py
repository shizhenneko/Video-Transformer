class KeyExhaustedError(Exception):
    """Exception raised when an API key is exhausted (e.g. too many 429s)."""
    pass
