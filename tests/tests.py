import unittest
from context import in3120


class TestInMemoryDictionary(unittest.TestCase):
    def test_access_vocabulary(self):
        vocabulary = in3120.InMemoryDictionary()
        vocabulary.add_if_absent("foo")
        vocabulary.add_if_absent("bar")
        vocabulary.add_if_absent("foo")
        self.assertEqual(len(vocabulary), 2)
        self.assertEqual(vocabulary.size(), 2)
        self.assertEqual(vocabulary.get_term_id("foo"), 0)
        self.assertEqual(vocabulary.get_term_id("bar"), 1)
        self.assertEqual(vocabulary["bar"], 1)
        self.assertIn("bar", vocabulary)
        self.assertNotIn("wtf", vocabulary)
        self.assertIsNone(vocabulary.get_term_id("wtf"))
        self.assertListEqual(sorted([v for v in vocabulary]), [("bar", 1), ("foo", 0)])


class TestBrainDeadNormalizer(unittest.TestCase):
    def setUp(self):
        self.__normalizer = in3120.BrainDeadNormalizer()

    def test_canonicalize(self):
        self.assertEqual(self.__normalizer.canonicalize("Dette ER en\nprØve!"), "Dette ER en\nprØve!")

    def test_normalize(self):
        self.assertEqual(self.__normalizer.normalize("grÅFustaSJE"), "gråfustasje")


class TestBrainDeadTokenizer(unittest.TestCase):
    def setUp(self):
        self.__tokenizer = in3120.BrainDeadTokenizer()

    def test_strings(self):
        result = list(self.__tokenizer.strings("Dette  er en\nprøve!"))
        self.assertListEqual(result, ["Dette", "er", "en", "prøve"])

    def test_tokens(self):
        result = list(self.__tokenizer.tokens("Dette  er en\nprøve!"))
        self.assertListEqual(result, [("Dette", (0, 5)), ("er", (7, 9)), ("en", (10, 12)), ("prøve", (13, 18))])

    def test_ranges(self):
        result = list(self.__tokenizer.ranges("Dette  er en\nprøve!"))
        self.assertListEqual(result, [(0, 5), (7, 9), (10, 12), (13, 18)])

    def test_empty_input(self):
        self.assertListEqual(list(self.__tokenizer.strings("")), [])
        self.assertListEqual(list(self.__tokenizer.tokens("")), [])
        self.assertListEqual(list(self.__tokenizer.ranges("")), [])

    def test_uses_yield(self):
        from types import GeneratorType
        for i in range(0, 5):
            buffer = "foo " * i
            self.assertIsInstance(self.__tokenizer.ranges(buffer), GeneratorType)
            self.assertIsInstance(self.__tokenizer.tokens(buffer), GeneratorType)
            self.assertIsInstance(self.__tokenizer.strings(buffer), GeneratorType)


class TestShingleGenerator(unittest.TestCase):
    def setUp(self):
        self.__tokenizer = in3120.ShingleGenerator(3)

    def test_strings(self):
        self.assertListEqual(list(self.__tokenizer.strings("")), [])
        self.assertListEqual(list(self.__tokenizer.strings("b")), ["b"])
        self.assertListEqual(list(self.__tokenizer.strings("ba")), ["ba"])
        self.assertListEqual(list(self.__tokenizer.strings("ban")), ["ban"])
        self.assertListEqual(list(self.__tokenizer.strings("bana")), ["ban", "ana"])
        self.assertListEqual(list(self.__tokenizer.strings("banan")), ["ban", "ana", "nan"])
        self.assertListEqual(list(self.__tokenizer.strings("banana")), ["ban", "ana", "nan", "ana"])

    def test_tokens(self):
        self.assertListEqual(list(self.__tokenizer.tokens("ba")), [("ba", (0, 2))])
        self.assertListEqual(list(self.__tokenizer.tokens("banan")),
                             [("ban", (0, 3)), ("ana", (1, 4)), ("nan", (2, 5))])

    def test_ranges(self):
        self.assertListEqual(list(self.__tokenizer.ranges("ba")), [(0, 2)])
        self.assertListEqual(list(self.__tokenizer.ranges("banan")), [(0, 3), (1, 4), (2, 5)])

    def test_uses_yield(self):
        import types
        for i in range(0, 5):
            buffer = "x" * i
            self.assertIsInstance(self.__tokenizer.ranges(buffer), types.GeneratorType)
            self.assertIsInstance(self.__tokenizer.tokens(buffer), types.GeneratorType)
            self.assertIsInstance(self.__tokenizer.strings(buffer), types.GeneratorType)

    def test_shingled_mesh_corpus(self):
        normalizer = in3120.BrainDeadNormalizer()
        corpus = in3120.InMemoryCorpus("../data/mesh.txt")
        index = in3120.InMemoryInvertedIndex(corpus, ["body"], normalizer, self.__tokenizer)
        engine = in3120.SimpleSearchEngine(corpus, index)
        tester = TestSimpleSearchEngine()
        tester._process_query_verify_matches("orGAnik kEMmistry", engine,
                                             {"match_threshold": 0.1, "hit_count": 10},
                                             (10, 8.0, [4408, 4410, 4411, 16980, 16981]))
        tester._process_query_verify_matches("synndrome", engine,
                                             {"match_threshold": 0.1, "hit_count": 10},
                                             (10, 7.0, [1275]))


class TestSieve(unittest.TestCase):
    def test_sifting(self):
        sieve = in3120.Sieve(3)
        sieve.sift(1.0, "one")
        sieve.sift(10.0, "ten")
        sieve.sift(9.0, "nine")
        sieve.sift(2.0, "two")
        sieve.sift(5.0, "five")
        sieve.sift(8.0, "eight")
        sieve.sift(7.0, "seven")
        sieve.sift(6.0, "six")
        sieve.sift(3.0, "three")
        sieve.sift(4.0, "four")
        self.assertListEqual(list(sieve.winners()), [(10.0, "ten"), (9.0, "nine"), (8.0, "eight")])

    def test_invalid_size(self):
        for i in [-1, 0]:
            with self.assertRaises(AssertionError):
                in3120.Sieve(i)

    def test_empty_sieve(self):
        sieve = in3120.Sieve(3)
        self.assertListEqual(list(sieve.winners()), [])


class TestInMemoryCorpus(unittest.TestCase):
    def test_access_documents(self):
        corpus = in3120.InMemoryCorpus()
        corpus.add_document(in3120.InMemoryDocument(0, {"body": "this is a Test"}))
        corpus.add_document(in3120.InMemoryDocument(1, {"title": "prØve", "body": "en to tre"}))
        self.assertEqual(corpus.size(), 2)
        self.assertListEqual([d.document_id for d in corpus], [0, 1])
        self.assertListEqual([corpus[i].document_id for i in range(0, corpus.size())], [0, 1])
        self.assertListEqual([corpus.get_document(i).document_id for i in range(0, corpus.size())], [0, 1])

    def test_load_from_file(self):
        corpus = in3120.InMemoryCorpus("../data/mesh.txt")
        self.assertEqual(corpus.size(), 25588)
        corpus = in3120.InMemoryCorpus("../data/cran.xml")
        self.assertEqual(corpus.size(), 1400)
        corpus = in3120.InMemoryCorpus("../data/docs.json")
        self.assertEqual(corpus.size(), 13)
        corpus = in3120.InMemoryCorpus("../data/imdb.csv")
        self.assertEqual(corpus.size(), 1000)


class TestInMemoryInvertedIndex(unittest.TestCase):
    def setUp(self):
        self.__normalizer = in3120.BrainDeadNormalizer()
        self.__tokenizer = in3120.BrainDeadTokenizer()

    def test_access_postings(self):
        corpus = in3120.InMemoryCorpus()
        corpus.add_document(in3120.InMemoryDocument(0, {"body": "this is a Test"}))
        corpus.add_document(in3120.InMemoryDocument(1, {"body": "test TEST prØve"}))
        index = in3120.InMemoryInvertedIndex(corpus, ["body"], self.__normalizer, self.__tokenizer)
        self.assertListEqual(list(index.get_terms("PRøvE wtf tesT")), ["prøve", "wtf", "test"])
        self.assertListEqual([(p.document_id, p.term_frequency) for p in index["prøve"]], [(1, 1)])
        self.assertListEqual([(p.document_id, p.term_frequency) for p in index.get_postings_iterator("wtf")], [])
        self.assertListEqual([(p.document_id, p.term_frequency) for p in index["test"]], [(0, 1), (1, 2)])
        self.assertEqual(index.get_document_frequency("wtf"), 0)
        self.assertEqual(index.get_document_frequency("prøve"), 1)
        self.assertEqual(index.get_document_frequency("test"), 2)

    def test_mesh_corpus(self):
        corpus = in3120.InMemoryCorpus("../data/mesh.txt")
        index = in3120.InMemoryInvertedIndex(corpus, ["body"], self.__normalizer, self.__tokenizer)
        self.assertEqual(len(list(index["hydrogen"])), 8)
        self.assertEqual(len(list(index["hydrocephalus"])), 2)

    def test_multiple_fields(self):
        document = in3120.InMemoryDocument(0, {
            'felt1': 'Dette er en test. Test, sa jeg. TEST!',
            'felt2': 'test er det',
            'felt3': 'test TEsT',
        })
        corpus = in3120.InMemoryCorpus()
        corpus.add_document(document)
        index = in3120.InMemoryInvertedIndex(corpus, ['felt1', 'felt3'], self.__normalizer, self.__tokenizer)
        posting = next(index.get_postings_iterator('test'))
        self.assertEqual(posting.document_id, 0)
        self.assertEqual(posting.term_frequency, 5)


class TestPostingsMerger(unittest.TestCase):
    def setUp(self):
        self.__merger = in3120.PostingsMerger()

    def test_empty_lists(self):
        posting = in3120.Posting(123, 4)
        self.assertListEqual(list(self.__merger.intersection(iter([]), iter([]))), [])
        self.assertListEqual(list(self.__merger.intersection(iter([]), iter([posting]))), [])
        self.assertListEqual(list(self.__merger.intersection(iter([posting]), iter([]))), [])
        self.assertListEqual(list(self.__merger.union(iter([]), iter([]))), [])
        self.assertListEqual([p.document_id for p in self.__merger.union(iter([]), iter([posting]))],
                             [posting.document_id])
        self.assertListEqual([p.document_id for p in self.__merger.union(iter([posting]), iter([]))],
                             [posting.document_id])

    def test_order_independence(self):
        postings1 = [in3120.Posting(1, 0), in3120.Posting(2, 0), in3120.Posting(3, 0)]
        postings2 = [in3120.Posting(2, 0), in3120.Posting(3, 0), in3120.Posting(6, 0)]
        result12 = list(map(lambda p: p.document_id, self.__merger.intersection(iter(postings1), iter(postings2))))
        result21 = list(map(lambda p: p.document_id, self.__merger.intersection(iter(postings2), iter(postings1))))
        self.assertListEqual(result12, [2, 3])
        self.assertListEqual(result12, result21)
        result12 = list(map(lambda p: p.document_id, self.__merger.union(iter(postings1), iter(postings2))))
        result21 = list(map(lambda p: p.document_id, self.__merger.union(iter(postings2), iter(postings1))))
        self.assertListEqual(result12, [1, 2, 3, 6])
        self.assertListEqual(result12, result21)

    def test_uses_yield(self):
        import types
        postings1 = [in3120.Posting(1, 0), in3120.Posting(2, 0), in3120.Posting(3, 0)]
        postings2 = [in3120.Posting(2, 0), in3120.Posting(3, 0), in3120.Posting(6, 0)]
        result1 = self.__merger.intersection(iter(postings1), iter(postings2))
        result2 = self.__merger.union(iter(postings1), iter(postings2))
        self.assertIsInstance(result1, types.GeneratorType, "Are you using yield?")
        self.assertIsInstance(result2, types.GeneratorType, "Are you using yield?")

    def __process_query_with_two_terms(self, corpus, index, query, operator, expected):
        terms = list(index.get_terms(query))
        postings = [index[terms[i]] for i in range(len(terms))]
        self.assertEqual(len(postings), 2)
        merged = operator(postings[0], postings[1])
        documents = [corpus[posting.document_id] for posting in merged]
        self.assertEqual(len(documents), len(expected))
        self.assertListEqual([d.document_id for d in documents], expected)

    def test_mesh_corpus(self):
        normalizer = in3120.BrainDeadNormalizer()
        tokenizer = in3120.BrainDeadTokenizer()
        corpus = in3120.InMemoryCorpus("../data/mesh.txt")
        index = in3120.InMemoryInvertedIndex(corpus, ["body"], normalizer, tokenizer)
        self.__process_query_with_two_terms(corpus, index, "HIV  pROtein", self.__merger.intersection,
                                            [11316, 11319, 11320, 11321])
        self.__process_query_with_two_terms(corpus, index, "water Toxic", self.__merger.union,
                                            [3078, 8138, 8635, 9379, 14472, 18572, 23234, 23985] +
                                            [i for i in range(25265, 25282)])


class TestTrie(unittest.TestCase):
    def test_access_nodes(self):
        tokenizer = in3120.BrainDeadTokenizer()
        root = in3120.Trie()
        root.add(["abba", "ørret", "abb", "abbab", "abbor"], tokenizer)
        self.assertFalse(root.is_final())
        self.assertIsNone(root.consume("snegle"))
        node = root.consume("ab")
        self.assertFalse(node.is_final())
        node = node.consume("b")
        self.assertTrue(node.is_final())
        self.assertEqual(node, root.consume("abb"))


class TestSuffixArray(unittest.TestCase):
    def setUp(self):
        self.__normalizer = in3120.BrainDeadNormalizer()
        self.__tokenizer = in3120.BrainDeadTokenizer()

    def __process_query_and_verify_winner(self, engine, query, winners, score):
        options = {"debug": False, "hit_count": 5}
        matches = list(engine.evaluate(query, options))
        if winners:
            self.assertGreaterEqual(len(matches), 1)
            self.assertLessEqual(len(matches), 5)
            self.assertIn(matches[0]["document"].document_id, winners)
            if score:
                self.assertEqual(matches[0]["score"], score)
        else:
            self.assertEqual(len(matches), 0)

    def test_cran_corpus(self):
        corpus = in3120.InMemoryCorpus("../data/cran.xml")
        engine = in3120.SuffixArray(corpus, ["body"], self.__normalizer, self.__tokenizer)
        self.__process_query_and_verify_winner(engine, "visc", [328], 11)
        self.__process_query_and_verify_winner(engine, "Of  A", [946], 10)
        self.__process_query_and_verify_winner(engine, "", [], None)
        self.__process_query_and_verify_winner(engine, "approximate solution", [159, 1374], 3)

    def test_memory_usage(self):
        import tracemalloc
        import inspect
        corpus = in3120.InMemoryCorpus()
        corpus.add_document(in3120.InMemoryDocument(0, {"a": "o  o\n\n\no\n\no", "b": "o o\no   \no"}))
        corpus.add_document(in3120.InMemoryDocument(1, {"a": "ba", "b": "b bab"}))
        corpus.add_document(in3120.InMemoryDocument(2, {"a": "o  o O o", "b": "o o"}))
        corpus.add_document(in3120.InMemoryDocument(3, {"a": "oO" * 10000, "b": "o"}))
        corpus.add_document(in3120.InMemoryDocument(4, {"a": "cbab o obab O ", "b": "o o " * 10000}))
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()
        engine = in3120.SuffixArray(corpus, ["a", "b"], self.__normalizer, self.__tokenizer)
        self.assertIsNotNone(engine)
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()
        for statistic in snapshot2.compare_to(snapshot1, "filename"):
            if statistic.traceback[0].filename == inspect.getfile(in3120.SuffixArray):
                self.assertLessEqual(statistic.size_diff, 2000000, "Memory usage seems excessive.")

    def test_multiple_fields(self):
        corpus = in3120.InMemoryCorpus()
        corpus.add_document(in3120.InMemoryDocument(0, {"field1": "a b c", "field2": "b c d"}))
        corpus.add_document(in3120.InMemoryDocument(1, {"field1": "x", "field2": "y"}))
        corpus.add_document(in3120.InMemoryDocument(2, {"field1": "y", "field2": "z"}))
        engine0 = in3120.SuffixArray(corpus, ["field1", "field2"], self.__normalizer, self.__tokenizer)
        engine1 = in3120.SuffixArray(corpus, ["field1"], self.__normalizer, self.__tokenizer)
        engine2 = in3120.SuffixArray(corpus, ["field2"], self.__normalizer, self.__tokenizer)
        self.__process_query_and_verify_winner(engine0, "b c", [0], 2)
        self.__process_query_and_verify_winner(engine0, "y", [1, 2], 1)
        self.__process_query_and_verify_winner(engine1, "x", [1], 1)
        self.__process_query_and_verify_winner(engine1, "y", [2], 1)
        self.__process_query_and_verify_winner(engine1, "z", [], None)
        self.__process_query_and_verify_winner(engine2, "z", [2], 1)

    def test_uses_yield(self):
        import types
        corpus = in3120.InMemoryCorpus()
        corpus.add_document(in3120.InMemoryDocument(0, {"a": "the foo bar"}))
        engine = in3120.SuffixArray(corpus, ["a"], self.__normalizer, self.__tokenizer)
        matches = engine.evaluate("foo", {})
        self.assertIsInstance(matches, types.GeneratorType, "Are you using yield?")


class TestStringFinder(unittest.TestCase):
    def setUp(self):
        self.__tokenizer = in3120.BrainDeadTokenizer()

    def __scan_buffer_verify_matches(self, finder, buffer, expected):
        matches = list(finder.scan(buffer))
        self.assertListEqual([m["match"] for m in matches], expected)

    def test_scan_matches_only(self):
        dictionary = in3120.Trie()
        dictionary.add(["romerike", "apple computer", "norsk", "norsk ørret", "sverige",
                        "ørret", "banan", "a", "a b"], self.__tokenizer)
        finder = in3120.StringFinder(dictionary, self.__tokenizer)
        self.__scan_buffer_verify_matches(finder,
                                          "en norsk     ørret fra romerike likte abba fra sverige",
                                          ["norsk", "norsk ørret", "ørret", "romerike", "sverige"])
        self.__scan_buffer_verify_matches(finder, "the apple is red", [])
        self.__scan_buffer_verify_matches(finder, "", [])
        self.__scan_buffer_verify_matches(finder,
                                          "apple computer banan foo sverige ben reddik fy fasan",
                                          ["apple computer", "banan", "sverige"])
        self.__scan_buffer_verify_matches(finder, "a a b", ["a", "a", "a b"])

    def test_scan_matches_and_ranges(self):
        dictionary = in3120.Trie()
        dictionary.add(["eple", "drue", "appelsin", "drue appelsin rosin banan papaya"], self.__tokenizer)
        finder = in3120.StringFinder(dictionary, self.__tokenizer)
        results = list(finder.scan("et eple og en drue   appelsin  rosin banan papaya frukt"))
        self.assertListEqual(results, [{'match': 'eple', 'range': (3, 7)},
                                       {'match': 'drue', 'range': (14, 18)},
                                       {'match': 'appelsin', 'range': (21, 29)},
                                       {'match': 'drue appelsin rosin banan papaya', 'range': (14, 49)}])

    def test_uses_yield(self):
        from types import GeneratorType
        trie = in3120.Trie()
        trie.add(["foo"], self.__tokenizer)
        finder = in3120.StringFinder(trie, self.__tokenizer)
        matches = finder.scan("the foo bar")
        self.assertIsInstance(matches, GeneratorType, "Are you using yield?")

    def test_mesh_terms_in_cran_corpus(self):
        mesh = in3120.InMemoryCorpus("../data/mesh.txt")
        cran = in3120.InMemoryCorpus("../data/cran.xml")
        trie = in3120.Trie()
        trie.add((d["body"] or "" for d in mesh), self.__tokenizer)
        finder = in3120.StringFinder(trie, self.__tokenizer)
        self.__scan_buffer_verify_matches(finder, cran[0]["body"], ["wing", "wing"])
        self.__scan_buffer_verify_matches(finder, cran[3]["body"], ["solutions", "skin", "friction"])
        self.__scan_buffer_verify_matches(finder, cran[1254]["body"], ["electrons", "ions"])


class TestSimpleSearchEngine(unittest.TestCase):
    def setUp(self):
        self.__normalizer = in3120.BrainDeadNormalizer()
        self.__tokenizer = in3120.BrainDeadTokenizer()

    def __process_two_term_query_verify_matches(self, query, engine, options, expected):
        ranker = in3120.BrainDeadRanker()
        hits, winners = expected
        matches = list(engine.evaluate(query, options, ranker))
        matches = [(m["score"], m["document"].document_id) for m in matches]
        self.assertEqual(len(matches), hits)
        for (score, winner) in matches[:len(winners)]:
            self.assertEqual(score, 2.0)
            self.assertIn(winner, winners)
        for (score, contender) in matches[len(winners):]:
            self.assertEqual(score, 1.0)

    def test_mesh_corpus(self):
        corpus = in3120.InMemoryCorpus("../data/mesh.txt")
        index = in3120.InMemoryInvertedIndex(corpus, ["body"], self.__normalizer, self.__tokenizer)
        engine = in3120.SimpleSearchEngine(corpus, index)
        query = "polluTION Water"
        self.__process_two_term_query_verify_matches(query, engine,
                                                     {"match_threshold": 0.1, "hit_count": 10},
                                                     (10, [25274, 25275, 25276]))
        self.__process_two_term_query_verify_matches(query, engine,
                                                     {"match_threshold": 1.0, "hit_count": 10},
                                                     (3, [25274, 25275, 25276]))

    def _process_query_verify_matches(self, query, engine, options, expected):
        from itertools import takewhile
        ranker = in3120.BrainDeadRanker()
        hits, score, winners = expected
        matches = list(engine.evaluate(query, options, ranker))
        matches = [(m["score"], m["document"].document_id) for m in matches]
        self.assertEqual(len(matches), hits)
        if matches:
            for i in range(1, hits):
                self.assertGreaterEqual(matches[i - 1][0], matches[i][0])
            if score:
                self.assertEqual(matches[0][0], score)
            if winners:
                top = takewhile(lambda m: m[0] == matches[0][0], matches)
                self.assertListEqual(winners, list(sorted([m[1] for m in top])))

    def test_synthetic_corpus(self):
        from itertools import product, combinations_with_replacement
        corpus = in3120.InMemoryCorpus()
        words = ("".join(term) for term in product("bcd", "aei", "jkl"))
        texts = (" ".join(word) for word in combinations_with_replacement(words, 3))
        for text in texts:
            corpus.add_document(in3120.InMemoryDocument(corpus.size(), {"a": text}))
        index = in3120.InMemoryInvertedIndex(corpus, ["a"], self.__normalizer, self.__tokenizer)
        engine = in3120.SimpleSearchEngine(corpus, index)
        epsilon = 0.0001
        self._process_query_verify_matches("baj BAJ    baj", engine,
                                           {"match_threshold": 1.0, "hit_count": 27},
                                           (27, 9.0, [0]))
        self._process_query_verify_matches("baj caj", engine,
                                           {"match_threshold": 1.0, "hit_count": 100},
                                           (27, None, None))
        self._process_query_verify_matches("baj caj daj", engine,
                                           {"match_threshold": 2/3 + epsilon, "hit_count": 100},
                                           (79, None, None))
        self._process_query_verify_matches("baj caj", engine,
                                           {"match_threshold": 2/3 + epsilon, "hit_count": 100},
                                           (100, 3.0, [0, 9, 207, 2514]))
        self._process_query_verify_matches("baj cek dil", engine,
                                           {"match_threshold": 1.0, "hit_count": 10},
                                           (1, 3.0, [286]))
        self._process_query_verify_matches("baj cek dil", engine,
                                           {"match_threshold": 1.0, "hit_count": 10},
                                           (1, None, None))
        self._process_query_verify_matches("baj cek dil", engine,
                                           {"match_threshold": 2/3 + epsilon, "hit_count": 80},
                                           (79, 3.0, [13, 26, 273, 286, 377, 3107, 3198]))
        self._process_query_verify_matches("baj xxx yyy", engine,
                                           {"match_threshold": 2/3 + epsilon, "hit_count": 100},
                                           (0, None, None))
        self._process_query_verify_matches("baj xxx yyy", engine,
                                           {"match_threshold": 2/3 - epsilon, "hit_count": 100},
                                           (100, None, None))

    def test_document_at_a_time_traversal_mesh_corpus(self):
        from typing import Iterator, List, Tuple, Set

        class AccessLoggedCorpus(in3120.Corpus):
            def __init__(self, wrapped: in3120.Corpus):
                self.__wrapped = wrapped
                self.__accesses = set()

            def __iter__(self):
                return iter(self.__wrapped)

            def size(self) -> int:
                return self.__wrapped.size()

            def get_document(self, document_id: int) -> in3120.Document:
                self.__accesses.add(document_id)
                return self.__wrapped.get_document(document_id)

            def get_history(self) -> Set[int]:
                return self.__accesses

        class AccessLoggedIterator(Iterator[in3120.Posting]):
            def __init__(self, term: str, accesses: List[Tuple[str, int]], wrapped: Iterator[in3120.Posting]):
                self.__term = term
                self.__accesses = accesses
                self.__wrapped = wrapped

            def __next__(self):
                posting = next(self.__wrapped)
                self.__accesses.append((self.__term, posting.document_id))
                return posting

        class AccessLoggedInvertedIndex(in3120.InvertedIndex):
            def __init__(self, wrapped: in3120.InvertedIndex):
                self.__wrapped = wrapped
                self.__accesses = []

            def get_terms(self, buffer: str) -> Iterator[str]:
                return self.__wrapped.get_terms(buffer)

            def get_postings_iterator(self, term: str) -> Iterator[in3120.Posting]:
                return AccessLoggedIterator(term, self.__accesses, self.__wrapped.get_postings_iterator(term))

            def get_document_frequency(self, term: str) -> int:
                return self.__wrapped.get_document_frequency(term)

            def get_history(self) -> List[Tuple[str, int]]:
                return self.__accesses

        corpus1 = in3120.InMemoryCorpus("../data/mesh.txt")
        corpus2 = AccessLoggedCorpus(corpus1)
        index = AccessLoggedInvertedIndex(in3120.InMemoryInvertedIndex(corpus1, ["body"],
                                                                       self.__normalizer, self.__tokenizer))
        engine = in3120.SimpleSearchEngine(corpus2, index)
        ranker = in3120.BrainDeadRanker()
        query = "Water  polluTION"
        options = {"match_threshold": 0.5, "hit_count": 1, "debug": False}
        matches = list(engine.evaluate(query, options, ranker))
        self.assertIsNotNone(matches)
        history = corpus2.get_history()
        self.assertListEqual(list(history), [25274])  # Only the document in the result set should be accessed.
        ordering1 = [('water', 3078),  # Document-at-a-time ordering if evaluated as "water pollution".
                     ('pollution', 788), ('pollution', 789), ('pollution', 790), ('pollution', 8079),
                     ('water', 8635),
                     ('pollution', 23837),
                     ('water', 9379), ('water', 23234), ('water', 25265),
                     ('pollution', 25274),
                     ('water', 25266), ('water', 25267), ('water', 25268), ('water', 25269), ('water', 25270),
                     ('water', 25271), ('water', 25272), ('water', 25273), ('water', 25274), ('water', 25275),
                     ('pollution', 25275),
                     ('water', 25276),
                     ('pollution', 25276),
                     ('water', 25277), ('water', 25278), ('water', 25279), ('water', 25280), ('water', 25281)]
        ordering2 = [('pollution', 788),  # Document-at-a-time ordering if evaluated as "pollution water".
                     ('water', 3078),
                     ('pollution', 789), ('pollution', 790), ('pollution', 8079),
                     ('water', 8635),
                     ('pollution', 23837),
                     ('water', 9379), ('water', 23234), ('water', 25265),
                     ('pollution', 25274),
                     ('water', 25266), ('water', 25267), ('water', 25268), ('water', 25269), ('water', 25270),
                     ('water', 25271), ('water', 25272), ('water', 25273), ('water', 25274),
                     ('pollution', 25275),
                     ('water', 25275),
                     ('pollution', 25276),
                     ('water', 25276), ('water', 25277), ('water', 25278), ('water', 25279), ('water', 25280),
                     ('water', 25281)]
        history = index.get_history()
        self.assertTrue(history == ordering1 or history == ordering2)  # Strict. Advanced implementations might fail.

    def test_uses_yield(self):
        import types
        corpus = in3120.InMemoryCorpus()
        corpus.add_document(in3120.InMemoryDocument(0, {"a": "foo bar"}))
        index = in3120.InMemoryInvertedIndex(corpus, ["a"], self.__normalizer, self.__tokenizer)
        engine = in3120.SimpleSearchEngine(corpus, index)
        ranker = in3120.BrainDeadRanker()
        matches = engine.evaluate("foo", {}, ranker)
        self.assertIsInstance(matches, types.GeneratorType, "Are you using yield?")


class TestBrainDeadRanker(unittest.TestCase):
    def setUp(self):
        self.__ranker = in3120.BrainDeadRanker()

    def test_term_frequency(self):
        self.__ranker.reset(21)
        self.__ranker.update("foo", 2, in3120.Posting(21, 4))
        self.__ranker.update("bar", 1, in3120.Posting(21, 3))
        self.assertEqual(self.__ranker.evaluate(), 11)
        self.__ranker.reset(42)
        self.__ranker.update("foo", 1, in3120.Posting(42, 1))
        self.__ranker.update("baz", 2, in3120.Posting(42, 2))
        self.assertEqual(self.__ranker.evaluate(), 5)

    def test_document_id_mismatch(self):
        self.__ranker.reset(21)
        with self.assertRaises(AssertionError):
            self.__ranker.update("foo", 1, in3120.Posting(42, 4))


class TestBetterRanker(unittest.TestCase):
    def setUp(self):
        normalizer = in3120.BrainDeadNormalizer()
        tokenizer = in3120.BrainDeadTokenizer()
        corpus = in3120.InMemoryCorpus()
        corpus.add_document(in3120.InMemoryDocument(0, {"title": "the foo", "static_quality_score": 0.9}))
        corpus.add_document(in3120.InMemoryDocument(1, {"title": "the foo", "static_quality_score": 0.2}))
        corpus.add_document(in3120.InMemoryDocument(2, {"title": "the foo foo", "static_quality_score": 0.2}))
        corpus.add_document(in3120.InMemoryDocument(3, {"title": "the bar"}))
        corpus.add_document(in3120.InMemoryDocument(4, {"title": "the bar bar"}))
        corpus.add_document(in3120.InMemoryDocument(5, {"title": "the baz"}))
        corpus.add_document(in3120.InMemoryDocument(6, {"title": "the baz"}))
        corpus.add_document(in3120.InMemoryDocument(7, {"title": "the baz baz"}))
        index = in3120.InMemoryInvertedIndex(corpus, ["title"], normalizer, tokenizer)
        self.__ranker = in3120.BetterRanker(corpus, index)

    def test_term_frequency(self):
        self.__ranker.reset(1)
        self.__ranker.update("foo", 1, in3120.Posting(1, 1))
        score1 = self.__ranker.evaluate()
        self.__ranker.reset(2)
        self.__ranker.update("foo", 1, in3120.Posting(2, 2))
        score2 = self.__ranker.evaluate()
        self.assertGreater(score1, 0.0)
        self.assertGreater(score2, 0.0)
        self.assertGreater(score2, score1)

    def test_document_id_mismatch(self):
        self.__ranker.reset(21)
        with self.assertRaises(AssertionError):
            self.__ranker.update("foo", 1, in3120.Posting(42, 4))

    def test_inverse_document_frequency(self):
        self.__ranker.reset(3)
        self.__ranker.update("the", 1, in3120.Posting(3, 1))
        self.assertAlmostEqual(self.__ranker.evaluate(), 0.0, 8)
        self.__ranker.reset(3)
        self.__ranker.update("bar", 1, in3120.Posting(3, 1))
        score1 = self.__ranker.evaluate()
        self.__ranker.reset(5)
        self.__ranker.update("baz", 1, in3120.Posting(5, 1))
        score2 = self.__ranker.evaluate()
        self.assertGreater(score1, 0.0)
        self.assertGreater(score2, 0.0)
        self.assertGreater(score1, score2)

    def test_static_quality_score(self):
        self.__ranker.reset(0)
        self.__ranker.update("foo", 1, in3120.Posting(0, 1))
        score1 = self.__ranker.evaluate()
        self.__ranker.reset(1)
        self.__ranker.update("foo", 1, in3120.Posting(1, 1))
        score2 = self.__ranker.evaluate()
        self.assertGreater(score1, 0.0)
        self.assertGreater(score2, 0.0)
        self.assertGreater(score1, score2)


class TestNaiveBayesClassifier(unittest.TestCase):
    def setUp(self):
        self.__normalizer = in3120.BrainDeadNormalizer()
        self.__tokenizer = in3120.BrainDeadTokenizer()

    def test_china_example_from_textbook(self):
        import math
        china = in3120.InMemoryCorpus()
        china.add_document(in3120.InMemoryDocument(0, {"body": "Chinese Beijing Chinese"}))
        china.add_document(in3120.InMemoryDocument(1, {"body": "Chinese Chinese Shanghai"}))
        china.add_document(in3120.InMemoryDocument(2, {"body": "Chinese Macao"}))
        not_china = in3120.InMemoryCorpus()
        not_china.add_document(in3120.InMemoryDocument(0, {"body": "Tokyo Japan Chinese"}))
        training_set = {"china": china, "not china": not_china}
        classifier = in3120.NaiveBayesClassifier(training_set, ["body"], self.__normalizer, self.__tokenizer)
        results = list(classifier.classify("Chinese Chinese Chinese Tokyo Japan"))
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["category"], "china")
        self.assertAlmostEqual(math.exp(results[0]["score"]), 0.0003, 4)
        self.assertEqual(results[1]["category"], "not china")
        self.assertAlmostEqual(math.exp(results[1]["score"]), 0.0001, 4)

    def __classify_buffer_and_verify_top_categories(self, buffer, classifier, categories):
        results = list(classifier.classify(buffer))
        self.assertListEqual([results[i]["category"] for i in range(0, len(categories))], categories)

    def test_language_detection_trained_on_some_news_corpora(self):
        training_set = {language: in3120.InMemoryCorpus(f"../data/{language}.txt")
                        for language in ["en", "no", "da", "de"]}
        classifier = in3120.NaiveBayesClassifier(training_set, ["body"], self.__normalizer, self.__tokenizer)
        self.__classify_buffer_and_verify_top_categories("Vil det riktige språket identifiseres? Dette er bokmål.",
                                                         classifier, ["no"])
        self.__classify_buffer_and_verify_top_categories("I don't believe that the number of tokens exceeds a billion.",
                                                         classifier, ["en"])
        self.__classify_buffer_and_verify_top_categories("De danske drenge drikker snaps!",
                                                         classifier, ["da"])
        self.__classify_buffer_and_verify_top_categories("Der Kriminalpolizei! Haben sie angst?",
                                                         classifier, ["de"])

    def test_uses_yield(self):
        import types
        corpus = in3120.InMemoryCorpus()
        corpus.add_document(in3120.InMemoryDocument(0, {"a": "the foo bar"}))
        training_set = {c: corpus for c in ["x", "y"]}
        classifier = in3120.NaiveBayesClassifier(training_set, ["a"], self.__normalizer, self.__tokenizer)
        matches = classifier.classify("urg foo the gog")
        self.assertIsInstance(matches, types.GeneratorType, "Are you using yield?")


if __name__ == '__main__':
    unittest.main(verbosity=2)
