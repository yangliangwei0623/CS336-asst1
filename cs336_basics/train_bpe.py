"""BPE tokenizer training.

A straightforward (but reasonably efficient) implementation of byte-level BPE
training, following the CS336 assignment 1 specification:

1. The vocabulary is initialized with the 256 possible byte values plus any
   special tokens.
2. The corpus is split on special tokens (which are never merged across), and
   each resulting document is pre-tokenized with the GPT-2 regex.
3. Each pre-token is represented as a sequence of single-byte tokens. We
   repeatedly count adjacent token pairs and merge the most frequent one,
   breaking ties by preferring the lexicographically greater pair, until the
   vocabulary reaches ``vocab_size``.
"""

from __future__ import annotations

import os
from collections import defaultdict

import regex as re

# GPT-2 pre-tokenizer pattern (from the GPT-2 / tiktoken implementations).
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _merge_word(
    word: tuple[bytes, ...],
    pair: tuple[bytes, bytes],
    new_token: bytes,
) -> tuple[bytes, ...]:
    """Return ``word`` with every occurrence of ``pair`` merged into ``new_token``.

    Occurrences are merged greedily from left to right.
    """
    merged: list[bytes] = []
    i = 0
    n = len(word)
    while i < n:
        if i < n - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            merged.append(new_token)
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer.

    Args:
        input_path: Path to the UTF-8 encoded training corpus.
        vocab_size: Desired final vocabulary size, including the 256 byte values
            and the special tokens.
        special_tokens: Strings that are never split or merged.

    Returns:
        vocab: Mapping from token ID to token bytes.
        merges: Ordered list of merged byte pairs, in order of creation.
    """
    # 1. Initialize the vocabulary with the 256 byte values and special tokens.
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    # 2. Read the corpus and split it on special tokens so that merges never
    #    cross a special-token boundary.
    with open(input_path, "rb") as f:
        text = f.read().decode("utf-8", errors="ignore")

    if special_tokens:
        split_pattern = "|".join(re.escape(token) for token in special_tokens)
        documents = re.split(split_pattern, text)
    else:
        documents = [text]

    # 3. Pre-tokenize each document and count pre-token frequencies. Each
    #    pre-token starts life as a tuple of single-byte tokens.
    word_freqs: dict[tuple[bytes, ...], int] = defaultdict(int)
    for document in documents:
        for match in re.finditer(PAT, document):
            encoded = match.group().encode("utf-8")
            word = tuple(bytes([b]) for b in encoded)
            word_freqs[word] += 1

    # 4. Build the initial pair statistics: how often each adjacent pair occurs,
    #    and which words contain each pair (so we only revisit affected words).
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
    for word, freq in word_freqs.items():
        for pair in zip(word, word[1:]):
            pair_counts[pair] += freq
            pair_to_words[pair].add(word)

    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)

    for _ in range(num_merges):
        if not pair_counts:
            break

        # Most frequent pair; ties broken by the lexicographically greater pair.
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))

        merges.append(best)
        new_token = best[0] + best[1]
        vocab[len(vocab)] = new_token

        # Only words that currently contain `best` can change. Snapshot the
        # affected words together with their frequencies before mutating.
        affected = [(word, word_freqs[word]) for word in pair_to_words[best]]

        # Remove each affected word's contribution to the statistics.
        for word, freq in affected:
            del word_freqs[word]
            for pair in zip(word, word[1:]):
                pair_counts[pair] -= freq
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                bucket = pair_to_words.get(pair)
                if bucket is not None:
                    bucket.discard(word)

        # Add the merged words back. A merged word may collide with an existing
        # entry (or another merged word); accumulating frequencies works because
        # pair contributions scale linearly with frequency.
        for word, freq in affected:
            merged_word = _merge_word(word, best, new_token)
            word_freqs[merged_word] += freq
            for pair in zip(merged_word, merged_word[1:]):
                pair_counts[pair] += freq
                pair_to_words[pair].add(merged_word)

    return vocab, merges
