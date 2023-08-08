import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import trainers

def bpe_encode(sequences):
    tokenizer = Tokenizer(BPE())
    corpus = [sequences]
    trainer = trainers.BpeTrainer(vocab_size=1000, min_frequency=2, special_tokens=[],
                                show_progress=True)

    tokenizer.train_from_iterator(corpus, trainer)

    encoded_sequence = tokenizer.encode(sequences)
    vocab = tokenizer.get_vocab()
    return encoded_sequence.tokens, vocab
