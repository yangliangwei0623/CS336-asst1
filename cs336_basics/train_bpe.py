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

import multiprocessing
import os
from collections import Counter, defaultdict
from typing import BinaryIO

import regex as re

# GPT-2 pre-tokenizer pattern (from the GPT-2 / tiktoken implementations).
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Below this file size (bytes) we pre-tokenize serially: spawning worker
# processes would cost more than it saves on small corpora.
_PARALLEL_MIN_BYTES = 1_000_000


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Split a file into byte ranges that can be pre-tokenized independently.

    Boundaries are snapped forward to the next occurrence of
    ``split_special_token`` so that no pre-token (and no special token) is cut in
    half across chunks. May return fewer than ``desired_num_chunks`` ranges if
    boundaries collapse onto each other.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    # Initial guesses for chunk boundary locations, uniformly spaced.
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead 4k bytes at a time.
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":  # EOF: snap this boundary to end of file.
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def _count_pretokens(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    """Pre-tokenize ``text`` and return pre-token frequencies.

    The text is split on special tokens (never merged across), then each document
    is pre-tokenized with the GPT-2 regex. Every pre-token is stored as a tuple
    of single-byte tokens.
    """
    if special_tokens:
        split_pattern = "|".join(re.escape(token) for token in special_tokens)
        documents = re.split(split_pattern, text)
    else:
        documents = [text]

    counts: Counter[tuple[bytes, ...]] = Counter()
    for document in documents:
        for match in re.finditer(PAT, document):
            encoded = match.group().encode("utf-8")
            word = tuple(bytes([b]) for b in encoded)
            counts[word] += 1
    return counts


def _pretokenize_chunk(args: tuple[str, int, int, list[str]]) -> Counter[tuple[bytes, ...]]:
    """Worker entry point: read a byte range from the file and count pre-tokens."""
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    return _count_pretokens(text, special_tokens)


def _parallel_pretokenize(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int | None,
) -> Counter[tuple[bytes, ...]]:
    """Pre-tokenize the whole corpus, in parallel for large files.

    The file is chunked at occurrences of the first special token so each worker
    processes a self-contained byte range; the per-chunk pre-token Counters are
    then summed into a single frequency table.
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    num_processes = max(1, num_processes)

    file_size = os.path.getsize(input_path)
    # Chunk boundaries must land on a byte string that actually occurs in the
    # file. A special token is ideal (it also bounds documents); fall back to a
    # newline when no special tokens were given.
    split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_token)

    chunk_args = [
        (os.fspath(input_path), start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    word_freqs: Counter[tuple[bytes, ...]] = Counter()
    # Small file, single worker, or a single chunk: skip process startup cost.
    if num_processes == 1 or len(chunk_args) <= 1 or file_size < _PARALLEL_MIN_BYTES:
        for args in chunk_args:
            word_freqs.update(_pretokenize_chunk(args))
    else:
        with multiprocessing.Pool(num_processes) as pool:
            for counts in pool.imap_unordered(_pretokenize_chunk, chunk_args):
                word_freqs.update(counts)

    return word_freqs


#返回合并后的word
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
    num_processes: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer.

    Args:
        input_path: Path to the UTF-8 encoded training corpus.
        vocab_size: Desired final vocabulary size, including the 256 byte values
            and the special tokens.
        special_tokens: Strings that are never split or merged.
        num_processes: Number of worker processes for pre-tokenization. Defaults
            to the CPU count; ``1`` forces serial pre-tokenization.

    Returns:
        vocab: Mapping from token ID to token bytes.
        merges: Ordered list of merged byte pairs, in order of creation.
    """
    # 1. Initialize the vocabulary with the 256 byte values and special tokens.
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    # 2 & 3. Pre-tokenize the corpus (parallelized for large files) into a table
    #        of pre-token -> frequency, where each pre-token is a tuple of
    #        single-byte tokens. Special tokens bound the chunks and documents,
    #        so merges never cross a special-token boundary.
    word_freqs = _parallel_pretokenize(input_path, special_tokens, num_processes)

    # 4. Build the initial pair statistics: how often each adjacent pair occurs,
    #    and which words contain each pair (so we only revisit affected words).
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)# 记录每个字节对（Pair）出现在了哪些单词（Word）中。
    for word, freq in word_freqs.items():
        for pair in zip(word, word[1:]):
            pair_counts[pair] += freq
            pair_to_words[pair].add(word)

    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)#vocab_size代表大模型最终能输出多少种token，每执行一次合并，能输出的token种类就在已有基础上

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


def _save_bpe(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    output_dir: str | os.PathLike,
) -> None:
    """Serialize the trained vocab and merges to ``output_dir``.

    ``vocab.pkl`` and ``merges.pkl`` hold the raw ``bytes`` objects (ready to feed
    straight back into a tokenizer). ``merges.txt`` is a human-readable dump.
    """
    import pickle

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    with open(os.path.join(output_dir, "merges.pkl"), "wb") as f:
        pickle.dump(merges, f)
    with open(os.path.join(output_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a!r} {b!r}\n")


def main() -> None:
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument("--input_path", required=True, help="Path to the training corpus.")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Final vocabulary size.")
    parser.add_argument(
        "--special_tokens",
        nargs="*",
        default=["<|endoftext|>"],
        help="Special tokens that are never split or merged.",
    )
    parser.add_argument("--output_dir", default=".", help="Directory to write vocab/merges into.")
    parser.add_argument(
        "--num_processes",
        type=int,
        default=None,
        help="Worker processes for pre-tokenization (default: CPU count).",
    )
    args = parser.parse_args()

    start = time.time()
    vocab, merges = train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        num_processes=args.num_processes,
    )
    elapsed = time.time() - start

    _save_bpe(vocab, merges, args.output_dir)
    longest = max(vocab.values(), key=len)
    print(f"Trained BPE in {elapsed:.1f}s: {len(vocab)} vocab, {len(merges)} merges.")
    print(f"Longest token ({len(longest)} bytes): {longest!r}")
    print(f"Saved vocab.pkl / merges.pkl / merges.txt to {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
