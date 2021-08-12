#!/usr/bin/python
# -*- coding: utf-8 -*-

import itertools
from collections import Counter
from .sieve import Sieve
from .corpus import Corpus
from .normalizer import Normalizer
from .tokenizer import Tokenizer
from typing import Any, Dict, Iterator, Iterable, Tuple


class SuffixArray:
    """
    A simple suffix array implementation. Allows us to conduct efficient substring searches.
    The prefix of a suffix is an infix!

    In a serious application we'd make use of least common prefixes (LCPs), pay more attention
    to memory usage, and add more lookup/evaluation features.
    """

    def __init__(self, corpus: Corpus, fields: Iterable[str], normalizer: Normalizer, tokenizer: Tokenizer):
        self.__corpus = corpus
        self.__haystack = []
        self.__suffixes = []
        self.__normalizer = normalizer
        self.__tokenizer = tokenizer
        self.__build_suffix_array(fields)

        self._document_contents = {}

    def __build_suffix_array(self, fields: Iterable[str]) -> None:
        """
        Builds a simple suffix array from the set of named fields in the document collection.
        The suffix array allows us to search across all named fields in one go.
        """
        raise NotImplementedError("You must implement _build_suffix_array method in SuffixArrray")

    def __normalize(self, buffer: str) -> str:
        """
        Produces a normalized version of the given string. Both queries and documents need to be
        identically processed for lookups to succeed.
        """
        # Tokenize and join to be robust to nuances in whitespace and punctuation.
        return self.__normalizer.normalize(" ".join(self.__tokenizer.strings(self.__normalizer.canonicalize(buffer))))

    def __binary_search(self, needle: str) -> int:
        """
        Does a binary search for a given normalized query (the needle) in the suffix array (the haystack).
        Returns the position in the suffix array where the normalized query is either found, or, if not found,
        should have been inserted.

        Kind of silly to roll our own binary search instead of using the bisect module, but seems needed
        due to how we represent the suffixes via (index, offset) tuples.
        """
        raise NotImplementedError("You must implement the __binary_serach method in SuffixArray")

    def evaluate(self, query: str, options: dict) -> Iterator[Dict[str, Any]]:
        """
        Evaluates the given query, doing a "phrase prefix search".  E.g., for a supplied query phrase like
        "to the be", we return documents that contain phrases like "to the bearnaise", "to the best",
        "to the behemoth", and so on. I.e., we require that the query phrase starts on a token boundary in the
        document, but it doesn't necessarily have to end on one.

        The matching documents are ranked according to how many times the query substring occurs in the document,
        and only the "best" matches are yielded back to the client. Ties are resolved arbitrarily.

        The client can supply a dictionary of options that controls this query evaluation process: The maximum
        number of documents to return to the client is controlled via the "hit_count" (int) option.

        The results yielded back to the client are dictionaries having the keys "score" (int) and
        "document" (Document).
        """
        raise NotImplementedError("You must implement the evaluate method in SuffixArray")
