#!/usr/bin/python
# -*- coding: utf-8 -*-

import itertools
from abc import ABC, abstractmethod
from .dictionary import InMemoryDictionary
from .normalizer import Normalizer
from .tokenizer import Tokenizer
from .corpus import Corpus
from .posting import Posting
from collections import Counter
from typing import Iterable, Iterator, List


class InvertedIndex(ABC):
    """
    Abstract base class for a simple inverted index.
    """

    def __getitem__(self, term: str) -> Iterator[Posting]:
        return self.get_postings_iterator(term)

    def __contains__(self, term: str) -> bool:
        return self.get_document_frequency(term) > 0

    @abstractmethod
    def get_terms(self, buffer: str) -> Iterator[str]:
        """
        Processes the given text buffer and returns an iterator that yields normalized
        terms as they are indexed. Both query strings and documents need to be
        identically processed.
        """
        pass

    @abstractmethod
    def get_postings_iterator(self, term: str) -> Iterator[Posting]:
        """
        Returns an iterator that can be used to iterate over the term's associated
        posting list. For out-of-vocabulary terms we associate empty posting lists.
        """
        pass

    @abstractmethod
    def get_document_frequency(self, term: str) -> int:
        """
        Returns the number of documents in the indexed corpus that contains the given term.
        """
        pass


class InMemoryInvertedIndex(InvertedIndex):
    """
    A simple in-memory implementation of an inverted index, suitable for small corpora.

    In a serious application we'd have configuration to allow for field-specific NLP,
    scale beyond current memory constraints, have a positional index, and so on.
    """

    def __init__(self, corpus: Corpus, fields: Iterable[str], normalizer: Normalizer, tokenizer: Tokenizer):
        self.__corpus = corpus
        self.__normalizer = normalizer
        self.__tokenizer = tokenizer
        self.__posting_lists : List[Posting] = []
        self.__dictionary = InMemoryDictionary()
        self.__build_index(fields)  # Constructs __posting_lists and __dictionary.

    def __repr__(self):
        return str({term: self.__posting_lists[term_id] for (term, term_id) in self.__dictionary})

    def __build_index(self, fields: Iterable[str]) -> None:
        """
        Builds a simple inverted index from the named fields in the document
        collection. The dictionary implementation is assumed to produce term
        identifiers in the range {0, ..., N - 1}.
        """
        raise NotImplementedError("You need to implement this as part of the assignment.")

    def get_terms(self, buffer: str) -> Iterator[str]:
        tokens = self.__tokenizer.strings(self.__normalizer.canonicalize(buffer))
        return (self.__normalizer.normalize(t) for t in tokens)

    def get_postings_iterator(self, term: str) -> Iterator[Posting]:
        # In a serious application a postings list would be stored as a contiguous buffer
        # storing compressed integers, and the iterator would facilitate loading this buffer
        # from somewhere and decompressing the integers.
        term_id = self.__dictionary.get_term_id(term)
        return iter([]) if term_id is None else iter(self.__posting_lists[term_id])

    def get_document_frequency(self, term: str) -> int:
        # In a serious application we'd store this number explicitly, e.g., as part of the dictionary.
        # That way, we can look up the document frequency without having to access the posting lists
        # themselves. Imagine if the posting lists don't even reside in memory!
        term_id = self.__dictionary.get_term_id(term)
        return 0 if term_id is None else len(self.__posting_lists[term_id])
