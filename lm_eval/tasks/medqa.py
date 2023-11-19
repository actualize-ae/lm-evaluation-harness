"""
MedQA: A Large-scale Open Domain Question Answering Dataset from Medical Exams

https://arxiv.org/abs/2009.13081

Open domain question answering (OpenQA) tasks have been recently attracting
more and more attention from the natural language processing (NLP) community.
In this work, we present the first free-form multiple-choice OpenQA dataset
for solving medical problems, MedQA, collected from the professional medical
board exams. It covers three languages: English, simplified Chinese, and traditional
Chinese, and contains 12,723, 34,251, and 14,123 questions for the three languages,
respectively. We implement both rule-based and popular neural methods by sequentially
combining a document retriever and a machine comprehension model. Through experiments,
we find that even the current best method can only achieve 36.7\%, 42.0\%, and 70.1\%
of test accuracy on the English, traditional Chinese, and simplified Chinese questions,
respectively. We expect MedQA to present great challenges to existing OpenQA systems and
hope that it can serve as a platform to promote much stronger OpenQA models from the NLP
community in the future.

Homepage: https://github.com/jind11/MedQA
"""
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@article{jin2020disease,
  title={What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams},
  author={Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung and Fang, Hanyi and Szolovits, Peter},
  journal={arXiv preprint arXiv:2009.13081},
  year={2020}
}
"""


class MedQA(MultipleChoiceTask):
    VERSION = 0
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "bigbio/med_qa"
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = "med_qa_en_4options_bigbio_qa"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        if self.has_test_docs():
            return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        choices = doc["choices"]

        return {
            "query": doc["question"],  # The query prompt.
            "choices": choices,  # The list of choices.
            "gold": doc["choices"].index(doc["answer"][0]),
        }

    def doc_to_text(self, doc):
        return (
                "You are a highly intelligent doctor who answers the following multiple choice question correctly.\nOnly write the answer down."
                + "\n\n**Question:**" + doc["query"] + "\n\n"
                + ",".join(doc['choices'])
                + "\n\n**Answer:**"
        )

    def doc_to_target(self, doc):
        return " " + doc["choices"][doc["gold"]] + "\n\n"
