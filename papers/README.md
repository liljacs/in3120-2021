# Papers

This folder contains various papers that supplement the [textbook](https://nlp.stanford.edu/IR-book/information-retrieval-book.html) in selected areas and/or provide additional detail to textbook topics. You will not be expected to know technical minutiae, but should be able to retell the gist of what those papers are about and their basic ideas.

Some of the papers below are key to the course assignments or are covered by lectures, and marked with (🌟). Others are recommended reading but already largely covered by the textbook or not crucial for the assignments, and marked with (🍀). Lastly, some papers are only intended as extra deep-dive reading for the interested student, and marked with (☕).

## Algorithms and data structures for search

* (🍀/☕) W. Pugh, ["Skip Lists: A Probabilistic Alternative to Balanced Trees"](skip-lists.pdf).

  The textbook discusses skip lists as a way to speed up the evaluation of posting list intersections in sublinear time, along with some heuristics for placing the skip pointers. This the the original paper on skip lists and goes into more depth on the skip list data structure, and discusses how multi-level skip lists (i.e., having skip lists on top of skip lists) become a tree-like data structure.

* (🍀) A. Z. Broder, D. Carmel, M. Herscovici, A. Soffer, J. Zien, ["Efficient Query Evaluation using a Two-Level Retrieval Process"](efficient-query-evaluation.pdf).

  In one of the programming assigments you are asked to implement efficient document-at-a-time retrieval over an inverted index to answer ranked "soft AND" queries. If you extrapolate that assigment slightly, you quickly end up with something similar to what this paper describes.

* (🍀) B. H. Bloom, ["Space/Time Trade-offs in Hash Coding with Allowable Errors"](space-time-trade-offs-in-hash-coding-with-allowable-errors.pdf).

  This is the original paper on Bloom filters. A Bloom filter is an extremely useful and space-efficient probabilistic data structure for use in search and elsewhere: It's a data structure that helps you answer set membership questions, without actually storing the set itself. Anwers may be wrong, but only in the false positive sense and not in the false negative sense: So no means no, and yes means maybe. You can control the probability of false positives through how you construct the Bloom filter.

* (☕) B. Fan, D. G. Andersen, M. Kaminsky, M. D. Mitzenmacher, ["Cuckoo Filter: Practically Better Than Bloom"](cuckoo-filter-practically-better-than-bloom.pdf).

  This is an example of a data structure in the Bloom filter family, that extends Bloom's original exposition. For example, it allows for deleting elements.

* (🍀) B. Goodwin, M. Hopcroft, D. Luu, A. Clemmer, M. Curmei, S. Elnikety, Y. He, ["BitFunnel: Revisiting Signatures for Search"](bitfunnel-revisiting-signatures-for-search.pdf).

  The textbook starts with discussing document-term matrices, and then arrives at an inverted index as a way to represent a sparse document-term matrix. Alternative approaches and representations exist, and one that has shown itself feasible and cost-efficient is to consider rows of bits (the "signature" of a document) in a document-term matrix as the primary way to represent an indexed document. This paper illustrates such an approach in use at Microsoft.

* (☕) Y. A. Malkov, D. A. Yashunin, ["Efficient and Robust Approximate Nearest Neighbor Search using Hierarchical Navigable Small World Graphs"](efficient-and-robust-approximate-nearest-neighbor-search.pdf).

  Through appropriate preprocessing we can represent documents and queries as dense numerical vectors, e.g., using the embedding vector techniques described in some of the other papers listed here. Searching for the most relevant documents then becomes a problem of being able to very efficiently identify which document vectors that are the closest to the query vector according to some suitable distance metric. This paper describes an index structure specialized for this purpose.

## Algorithms and data structures for strings

* (🍀) U. Germann, E. Joanis, S. Larkin, ["Tightly Packed Tries: How to Fit Large Models in Memory and Make them Load Fast, Too"](tightly-packed-tries.pdf).

  The need to compactly and efficiently represent dictionaries, i.e., large sets of strings, shows up in a number of places in a search engine. You often want to store them in a trie-like data structure to facilitate prefix searches and other efficient lookup strategies. Implementation-wise, in practice you do not represent large tries in memory using separate node objects and traditional pointers, but instead you pack everything into a big, contiguous byte buffer that you can jump around in. This paper illustrates one way to do this.

* (☕) M. Ciura, S. Deorowicz, ["How to Squeeze a Lexicon"](how-to-squeeze-a-lexicon.pdf).

  A trie shares prefixes among strings, but for extra oomph and memory savings you will often want to share suffixes as well. This paper shows how you can do this, effectively transforming your trie into a finite state automaton.

* (🌟) U. Manber, G. Myers, ["Suffix Arrays: A New Method for Online String Searches"](suffix-arrays.pdf).

  This paper shows how you can do efficient (say, as you type) substring searches in long strings or large dictionaries, by creating a suitably sorted array of positional indexes that you do binary search over. Based on the insight that a substring is a prefix of a suffix.

* (🌟) H. Shang, T. H. Merrett, ["Tries for Approximate String Matching"](tries-for-approximate-string-matching.pdf).

  The textbook discusses computing the edit distance between two strings, but not how to efficiently compute the edit distance between a string and a large set of strings. Imagine for example that you need to quickly spellcheck a search query by comparing it to a huge dictionary of known correct spellings: This paper shows how you can represent your dictionary as a trie, and then efficiently traverse this trie while updating an edit table along the way to locate the "best" matches in the dictionary.

* (☕) H. Hyyrö, ["Bit-Parallel Approximate String Matching Algorithms with Transposition"](bit-parallel-approximate-string-matching.pdf).

  The textbook discusses the concept of an edit table and the rules for updating this table. Assuming unit edit costs, this paper shows how you can achieve a speedup proportional to the computer's word size, by cleverly representing the edit table as a collection of bit vectors and then realizing the update rules through a series of complex combinations and manipulations of these bit vectors. For example, on a 64-bit computer, you would update blocks of 64 table cells in one swoop.

* (🌟) A. V. Aho, M. J. Corasick, ["Efficient String Matching: An Aid to Bibliographic Search"](efficient-string-matching.pdf).

  A common application in query- and content processing is to scan through a text buffer looking for all occurrences of strings that also appear in a large dictionary. In a sense, we want to compute the "intersection" between the text buffer and the dictionary. By representing your dictionary as a trie (with some extra edges), this paper tells you how you can do this very efficiently in a way that in practice is virtually insensitive to the size of your dictionary. In one of the programming assignments you will be asked to implement a trie-walk similar to this, with some simplifications and minor NLP extensions.

## Compression

* (🍀) D. Lemire, L. Boytsov, ["Decoding Billions of Integers Per Second Through Vectorization"](decoding-billions-of-integers-per-second.pdf).

  The posting lists in an inverted index are typically compressed to yield a number of performance benefits, and the efficiency of decompressing these lists is thus a focus area. Industrial-strength implementations are heavily optimized and concerned with utilizing the available hardware to its fullest, as this paper exemplifies. The paper's Related Work section gives a good summary of several families of relevant compression algorithms, including Simple-9 and PFOR which are not covered by the textbook. The hardware- and implementation focus of the paper is outside the scope of this course.

## Systems

* (🍀/☕) J. Dean, S. Ghemawat, ["MapReduce: Simplified Data Processing on Large Clusters"](mapreduce-simplified-data-processing-on-large-clusters.pdf).

  The textbook thoroughly discusses MapReduce in the context of distributed indexing. This is the original paper. A typical use of such systems is to process large logs and data sets, e.g., query- or clickthrough logs. Apache Hadoop, Apache Spark and similar systems are popular open-source systems inspired by MapReduce.

* (☕) R. Chaiken, B. Jenkins, P.-Å. Larson, B. Ramsey, D. Shakib, S. Weaver, J. Zhou, ["SCOPE: Easy and Efficient Parallel Processing of Massive Data Sets"](scope-easy-and-efficient-parallel-processing.pdf).

  This paper outlines a system similar to MapReduce, with jobs expressed in a SQL-like language. Used internally at Microsoft.

## Machine learning

* (☕) C. Burges, ["From RankNet to LambdaRank to LambdaMART: An Overview"](from-ranknet-to-lambdarank-to-lambdamart.pdf).

  The textbook discusses the concept of "learning to rank", i.e., looking at document ranking as an ML problem. All search engines today use ML models for ranking. This paper provides an overview of the problem statement and dives into some specific examples. 

* (🌟) A. Øhrn, ["Classifier Evaluation"](classifier-evaluation.pdf).

  When developing supervised ML models it is important that we do not "cheat" by evaluating the model on data it had available during training, so that we can assess how good it is at generalizing to new and unseen examples. The textbook discusses this, and this supplementary note goes into more detail and also covers topics such as cross-validation and ROC analysis.

* (🍀) Y. Goldberg, ["A Primer on Neural Network Models for Natural Language Processing"](a-primer-on-neural-network-models-for-natural-language-processing.pdf).

  Neural network models are all the rage in NLP these days. The first part of this paper provides a good overview of what neural networks are, how they are applied to text, and how training a model amounts to optimizing an objective function through gradient descent.

* (🍀) T. Mikolov, I. Sutskever, K. Chen, G. Corrado, J. Dean, ["Distributed Representations of Words and Phrases and their Compositionality"](distributed-representations-of-words-and-phrases-and-their-compositionality.pdf).

  The textbook discusses vector space models with one dimension per word or dictionary entry, which is a sparse high-dimensional representation. A more recent concept that has proven itself to be an important part of neural approaches to NLP is that of embedding vectors, where words and phrases are represented through dense lower-dimensional vectors. This paper outlines one approach for learning such embedding vectors. An important appeal of embedding vectors is that proximity in vector space also convey semantic similarity, enabling word analogies to often be solved with vector arithmetic. E.g., "queen ≈ king - man + woman".
