import logging
from typing import Dict

import tensorflow as tf

import global_const as glob_cont
from dataset_format_transformer.article_reference_transformer import ArticleReferenceTransformer

tf.compat.v1.enable_eager_execution()
logging.basicConfig(level=logging.INFO)


class ArticleReferenceTFRecordsTransformer(ArticleReferenceTransformer):

    def _save(self, storage: Dict[str, Dict[str, str]]):
        logging.info("Saving tf_record")
        with tf.io.TFRecordWriter(self.destination_path) as tfwriter:
            for file_key, desc_dict in storage.items():
                feature_dict = {
                    glob_cont.TF_RECORD_LABEL: self._bytes_feature(file_key),
                    glob_cont.TF_RECORD_ARTICLE: self._bytes_feature(desc_dict[glob_cont.TF_RECORD_ARTICLE]),
                    glob_cont.TF_RECORD_SUMMARY: self._bytes_feature(desc_dict[glob_cont.TF_RECORD_SUMMARY])
                }
                example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
                tfwriter.write(example_proto.SerializeToString())

    def _bytes_feature(self, value):
        """Преобразует string / byte в bytes_list."""
        value = value.encode("utf-8")
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == "__main__":
    article_folder = "/Users/denis/MyProjects/Summarization/data/BBC News Summary/News Articles"
    summary_folder = "/Users/denis/MyProjects/Summarization/data/BBC News Summary/Summaries"

    destination_folder = "/Users/denis/MyProjects/Summarization/data/tf_records/tf_records_article_reference/001.tfrecord"
    generator = ArticleReferenceTFRecordsTransformer(article_folder, summary_folder, destination_folder)
    generator.prepare_data()

    filenames = [destination_folder]
    raw_dataset = tf.data.TFRecordDataset(filenames)

    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        # example = tf.io.parse_single_example(serialized_example, feature_description)
        features = tf.parse_single_example(
            raw_record.numpy(),
            # Defaults are not specified since both keys are required.
            features={
                glob_cont.TF_RECORD_LABEL: tf.io.FixedLenFeature((), tf.string),
                glob_cont.TF_RECORD_ARTICLE: tf.io.FixedLenFeature((), tf.string),
                glob_cont.TF_RECORD_SUMMARY: tf.io.FixedLenFeature((), tf.string),
            })

        print(example)
        break
