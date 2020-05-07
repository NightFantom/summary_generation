from typing import List

from bert_experimental.feature_extraction.bert_feature_extractor import BERTFeatureExtractor
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np


class TreeRankSummariser:

    def __init__(self, max_sentence_int, encoder: BERTFeatureExtractor):
        self.encoder = encoder
        self.max_sentence_int = max_sentence_int

    def get_summary(self, text: str) -> List[str]:
        sentence_list = sent_tokenize(text)
        return self.get_summary_by_sentence(sentence_list, None)

    def get_summary_by_sentence(self, sentence_list: List[str], personalized) -> List[str]:
        vectors_np = self.encoder(sentence_list)
        similarity_matrix = cosine_similarity(vectors_np, vectors_np)
        for pos in range(similarity_matrix.shape[0]):
            similarity_matrix[pos][pos] = 0
        nx_graph = nx.from_numpy_array(similarity_matrix)
        pagerank_weights = nx.pagerank_numpy(nx_graph,personalization=personalized)

        sentence_np = []
        score_np = []
        for sentence_index, score in pagerank_weights.items():
            sentence_np.append(sentence_list[sentence_index])
            score_np.append(score)

        sentence_np = np.array(sentence_np)
        score_np = np.array(score_np)

        best_scores_np = score_np.argsort()[-self.max_sentence_int:]
        # Make extracted sentences ordered
        best_scores_np.sort()
        return list(sentence_np[best_scores_np])

