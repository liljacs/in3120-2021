#!/usr/bin/python
# -*- coding: utf-8 -*-

from .tokenizer import Tokenizer
from typing import Iterator, Tuple


class ShingleGenerator(Tokenizer):
    """
    Tokenizes a buffer into overlapping shingles having a specified width.
    """

    def __init__(self, width: int):
        assert width > 0
        self.__width = width

    def ranges(self, buffer: str) -> Iterator[Tuple[int, int]]:
        """
        Locates where the shingles begin and end. If the buffer is shorter than the shingle width
        then this produces a single shorter-than-usual shingle.

        The current implementation is simplistic and not whitespace- or punctuation-aware,
        and doesn't treat the beginning or end of the buffer in a special way.
        """
        raise NotImplementedError()
