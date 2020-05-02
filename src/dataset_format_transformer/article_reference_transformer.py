import logging
import os
from collections import defaultdict
from typing import Dict

import magic
import yaml

import global_const as glob_cont

logging.basicConfig(level=logging.INFO)


class ArticleReferenceTransformer:

    def __init__(self, article_folder: str, summary_folder: str, destination_path: str):
        self.article_folder = article_folder
        self.summary_folder = summary_folder
        self.destination_path = destination_path

    def _load_files(self, storage: Dict[str, Dict[str, str]], key: str, folder: str):
        for root, dirs, files in os.walk(folder):
            category = root.split("/")[-1]
            for file in files:
                file_key = f"{category}_{file}"
                path = os.path.join(root, file)
                extension = os.path.basename(path).split(".")[-1]
                if extension == "txt":
                    logging.info(f"Reading {path}")
                    encoder_determiner = magic.Magic(mime_encoding=True)
                    encoding = encoder_determiner.from_file(path)
                    with open(path, encoding=encoding) as file_source:
                        article = file_source.read()
                    storage[file_key][key] = article

    def _prepare_bundle(self):
        pairs_dict = defaultdict(lambda: {glob_cont.TF_RECORD_ARTICLE: None, glob_cont.TF_RECORD_SUMMARY: None})
        logging.info("Reading articles")
        self._load_files(pairs_dict, glob_cont.TF_RECORD_ARTICLE, self.article_folder)
        logging.info("Reading summaries")
        self._load_files(pairs_dict, glob_cont.TF_RECORD_SUMMARY, self.summary_folder)
        return pairs_dict

    def _save(self, storage: Dict[str, Dict[str, str]]):
        temp_result = []

        for key, val in storage.items():
            temp_result.append({
                glob_cont.TF_RECORD_SUMMARY: val[glob_cont.TF_RECORD_SUMMARY],
                glob_cont.TF_RECORD_ARTICLE: val[glob_cont.TF_RECORD_ARTICLE],
                glob_cont.TF_RECORD_LABEL: key
            })

        with open(self.destination_path, mode="w") as file:
            yaml.dump(temp_result, file)

    def prepare_data(self):
        pairs_dict = self._prepare_bundle()
        self._save(pairs_dict)


if __name__ == "__main__":
    article_folder = "/Users/denis/MyProjects/Summarization/data/BBC News Summary/News Articles"
    summary_folder = "/Users/denis/MyProjects/Summarization/data/BBC News Summary/Summaries"

    destination_folder = "/Users/denis/MyProjects/Summarization/data/yaml_format/article_summary_dataset.yml"
    generator = ArticleReferenceTransformer(article_folder, summary_folder, destination_folder)
    generator.prepare_data()