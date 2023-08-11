

class CharacterTokenizer:

    def __init__(self, characters):
        self._vocab_str_to_int = {
            "[UNK]": 0,
            **{ch: i+1 for i, ch in enumerate(characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

    def tokenize(self, text):
        tokens = []
        for char in text:
            if char not in self._vocab_str_to_int:
                tokens.append(self._vocab_str_to_int["UNK"])
            else:
                tokens.append(self._vocab_str_to_int[char])
        return tokens