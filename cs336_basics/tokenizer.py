"""Byte-level BPE tokenizer (encoding and decoding).

Given a vocabulary (``dict[int, bytes]``) and an ordered list of merges
(``list[tuple[bytes, bytes]]``), this tokenizer encodes text into integer token
IDs and decodes IDs back into text, matching the behaviour of the BPE trainer in
:mod:`cs336_basics.train_bpe`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator

import regex as re

# GPT-2 pre-tokenizer pattern (identical to the one used during training).
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    """A byte-level BPE tokenizer.

    Attributes:
        vocab: Mapping from token ID to token bytes.
        merges: BPE merges in order of creation (earlier = higher priority).
        special_tokens: Strings kept as single, indivisible tokens.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens) if special_tokens else []

        # Inverse vocabulary for looking up the ID of a byte sequence.
        self._token_to_id: dict[bytes, int] = {token: idx for idx, token in self.vocab.items()}

        # Merge priority: lower rank means the merge was created earlier and so
        # should be applied first when encoding.
        self._merge_rank: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }

        # Register special tokens, appending any that are missing from the vocab.
        self._special_token_to_id: dict[str, int] = {}
        for token in self.special_tokens:
            encoded = token.encode("utf-8")
            if encoded not in self._token_to_id:
                new_id = max(self.vocab) + 1 if self.vocab else 0
                self.vocab[new_id] = encoded
                self._token_to_id[encoded] = new_id
            self._special_token_to_id[token] = self._token_to_id[encoded]

        # Pre-build a regex that splits text on special tokens while keeping
        # them (the capturing group). Longer tokens come first so that, e.g.,
        # "<|eot|><|eot|>" is preferred over "<|eot|>" when both are special.
        if self.special_tokens:
            ordered = sorted(self.special_tokens, key=len, reverse=True)
            self._special_split_re = re.compile("(" + "|".join(re.escape(t) for t in ordered) + ")")
        else:
            self._special_split_re = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """Construct a Tokenizer from serialized vocab and merges files.

        Two on-disk formats are supported:

        * Pickle (``.pkl``): a pickled ``dict[int, bytes]`` vocab and a pickled
          ``list[tuple[bytes, bytes]]`` of merges (the format produced by the
          trainer's optional ``_save_bpe`` helper).
        * GPT-2 style: a JSON vocab mapping the GPT-2 unicode-escaped token
          strings to IDs, and a merges text file with one space-separated pair
          per line.
        """
        if str(vocab_filepath).endswith(".pkl"):
            import pickle

            with open(vocab_filepath, "rb") as f:
                vocab = pickle.load(f)
            with open(merges_filepath, "rb") as f:
                merges = pickle.load(f)
            return cls(vocab, merges, special_tokens)

        # GPT-2 style JSON vocab + text merges.
        byte_decoder = {ch: b for b, ch in _gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = {
            idx: bytes([byte_decoder[ch] for ch in token])
            for token, idx in raw_vocab.items()
        }
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                cleaned = line.rstrip("\n")
                parts = cleaned.split(" ")
                if len(parts) != 2:
                    continue
                left, right = parts
                merges.append(
                    (
                        bytes([byte_decoder[ch] for ch in left]),
                        bytes([byte_decoder[ch] for ch in right]),
                    )
                )
        return cls(vocab, merges, special_tokens)

    def _bpe_encode_chunk(self, text: str) -> list[int]:
        """BPE-encode a single pre-token string into token IDs."""
        # Start from the raw bytes, each as its own single-byte token.
        parts: list[bytes] = [bytes([b]) for b in text.encode("utf-8")]

        # Repeatedly merge the adjacent pair with the highest priority (lowest
        # merge rank) until no more merges apply.
        while len(parts) >= 2:
            best_rank: int | None = None
            best_i = -1
            for i in range(len(parts) - 1):
                rank = self._merge_rank.get((parts[i], parts[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_i = i
            if best_rank is None:
                break
            parts[best_i : best_i + 2] = [parts[best_i] + parts[best_i + 1]]

        return [self._token_to_id[part] for part in parts]

    def encode(self, text: str) -> list[int]:
        """Encode ``text`` into a list of token IDs."""
        if self._special_split_re is None:
            segments = [text]
        else:
            # Splitting on a capturing group keeps the special tokens themselves
            # as separate segments.
            segments = self._special_split_re.split(text)

        ids: list[int] = []
        for segment in segments:
            if not segment:
                continue
            special_id = self._special_token_to_id.get(segment)
            if special_id is not None:
                ids.append(special_id)
            else:
                for match in re.finditer(PAT, segment):
                    ids.extend(self._bpe_encode_chunk(match.group()))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode an iterable of strings (e.g. a file handle).

        Yields token IDs one at a time so that arbitrarily large streams can be
        tokenized with constant memory.
        """
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into text.

        Unknown IDs are skipped; any resulting malformed UTF-8 is replaced with
        the Unicode replacement character (U+FFFD).
        """
        token_bytes = b"".join(self.vocab[i] for i in ids if i in self.vocab)
        return token_bytes.decode("utf-8", errors="replace")


def _gpt2_bytes_to_unicode() -> dict[int, str]:
    """Reversible byte <-> printable-unicode mapping used by GPT-2's vocab files."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}
