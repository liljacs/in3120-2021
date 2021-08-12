#!/usr/bin/python
# -*- coding: utf-8 -*-

from abc import abstractmethod
import collections.abc
from .document import Document, InMemoryDocument


class Corpus(collections.abc.Iterable):
    """
    Abstract base class representing a corpus we can index and search over,
    i.e., a collection of documents. The class facilitates iterating over
    all documents in the corpus.
    """

    def __len__(self):
        return self.size()

    def __getitem__(self, document_id: int) -> Document:
        return self.get_document(document_id)

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def size(self) -> int:
        """
        Returns the size of the corpus, i.e., the number of documents in the
        document collection.
        """
        pass

    @abstractmethod
    def get_document(self, document_id: int) -> Document:
        """
        Returns the document associated with the given document identifier.
        """
        pass


class InMemoryCorpus(Corpus):
    """
    An in-memory implementation of a document store, suitable only for small
    document collections.

    Document identifiers are assigned on a first-come first-serve basis.
    """

    def __init__(self, filename=None):
        self._documents = []
        if filename:
            if filename.endswith(".txt"):
                self.__load_text(filename)
            elif filename.endswith(".xml"):
                self.__load_xml(filename)
            elif filename.endswith(".json"):
                self.__load_json(filename)
            elif filename.endswith(".csv"):
                self.__load_csv(filename)
            else:
                raise IOError("Unsupported extension")

    def __iter__(self):
        return iter(self._documents)

    def size(self) -> int:
        return len(self._documents)

    def get_document(self, document_id: int) -> Document:
        assert 0 <= document_id < len(self._documents)
        return self._documents[document_id]

    def add_document(self, document: Document) -> None:
        """
        Adds the given document to the corpus. Facilitates testing.
        """
        assert document.document_id == len(self._documents)
        self._documents.append(document)

    def __load_text(self, filename):
        """
        Loads documents from the given UTF-8 encoded text file. One document per line,
        tab-separated fields. Empty lines are ignored. The first field gets named "body",
        the second field (optional) gets named "meta". All other fields are currently ignored.
        """
        document_id = 0
        with open(filename, mode="r", encoding="utf-8") as f:
            for line in f:
                anonymous_fields = line.strip().split("\t")
                if len(anonymous_fields) == 1 and not anonymous_fields[0]:
                    continue
                named_fields = {"body": anonymous_fields[0]}
                if len(anonymous_fields) >= 2:
                    named_fields["meta"] = anonymous_fields[1]
                self.add_document(InMemoryDocument(document_id, named_fields))
                document_id += 1

    def __load_xml(self, filename):
        """
        Loads documents from the given XML file. The schema is assumed to be
        simple <doc> nodes. Each <doc> node gets mapped to a single document field
        named "body".
        """
        from xml.dom.minidom import parse

        def __get_text(nodes):
            data = []
            for node in nodes:
                if node.nodeType == node.TEXT_NODE:
                    data.append(node.data)
            return " ".join(data)

        dom = parse(filename)
        document_id = 0
        for body in [__get_text(n.childNodes) for n in dom.getElementsByTagName("doc")]:
            self.add_document(InMemoryDocument(document_id, {"body": body}))
            document_id += 1

    def __load_csv(self, filename):
        """
        Loads documents from the given UTF-8 encoded CSV file. One document per line.
        """
        import csv
        document_id = 0
        with open(filename, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.add_document(InMemoryDocument(document_id, dict(row)))
                document_id += 1

    def __load_json(self, filename):
        """
        Loads documents from the given UTF-8 encoded JSON file. One document per line.
        Lines that do not start with "{" are ignored.
        """
        from json import loads
        document_id = 0
        with open(filename, mode="r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("{"):
                    named_fields = loads(line)
                    self.add_document(InMemoryDocument(document_id, named_fields))
                    document_id += 1
