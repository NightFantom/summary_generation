import logging
from typing import List, Any, Dict

from bert_experimental.feature_extraction.bert_feature_extractor import BERTFeatureExtractor
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
import re
import global_const


class TreeRankSummariser:

    def __init__(self, max_sentence: int, encoder: BERTFeatureExtractor):
        self.encoder = encoder
        self.max_sentence_int = max_sentence
        self.__empty_list = []
        self.__logger = logging.getLogger(__name__)
        self.__stop_symbols = "[\.,\":'!?()\s]"

    def get_summary(self, text: str) -> List[str]:
        sentence_list = sent_tokenize(text)
        self.__logger.debug(f"Extracted {len(sentence_list)} sentences")
        summary = self.get_summary_by_sentence(sentence_list, None)
        return summary

    def get_summary_for_companies(self, text: str, companies: List[Dict[str, Any]]) -> Dict[str, str]:
        sentence_list = sent_tokenize(text)
        self.__logger.debug(f"Extracted {len(sentence_list)} sentences")
        isin_summary: Dict[str, str] = {}
        for company_dict in companies:
            isin_str = company_dict[global_const.COMPANY_ISIN]

            company_names_list = company_dict[global_const.COMPANY_NAMES]
            company_names_list = [el.lower() for el in company_names_list]
            company_names_str = "|".join(company_names_list)

            pattern_template = f"(^|{self.__stop_symbols})({company_names_str})($|{self.__stop_symbols})"
            self.__logger.debug(f"Pattern: {pattern_template}")
            pattern = re.compile(pattern_template)

            personalization = {}
            for index, sentence in enumerate(sentence_list):
                point = 0.1
                sentence = sentence.lower()
                result = pattern.search(sentence)
                if result is not None:
                    self.__logger.debug(f"Sentence {index} got weight. ISIN - {isin_str}")
                    self.__logger.debug(f"{sentence}")
                    point = 0.2
                personalization[index] = point

            summary = self.get_summary_by_sentence(sentence_list, personalization)
            summary_str = "★".join(summary)
            isin_summary[isin_str] = summary_str
        return isin_summary

    def get_summary_by_sentence(self, sentence_list: List[str], personalized) -> List[str]:
        vectors_np = self.encoder(sentence_list)
        similarity_matrix = cosine_similarity(vectors_np, vectors_np)
        for pos in range(similarity_matrix.shape[0]):
            similarity_matrix[pos][pos] = 0
        nx_graph = nx.from_numpy_array(similarity_matrix)
        pagerank_weights = nx.pagerank_numpy(nx_graph, personalization=personalized)

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

