# TODO: Remove all TODO comments once the implementation is complete.
"""
TODO: Add the Paper Title on this line.
TODO: Add the paper's PDF URL (preferably from arXiv) on this line.

TODO: Write a Short Description of the task.

Homepage: TODO: Add the URL to the task's Homepage here.
"""
from lm_eval.base import MultipleChoiceTask


# TODO: Add the BibTeX citation for the task.
_CITATION = """
"""


class MedQA_USMLE_ASSESSMENT(MultipleChoiceTask):
    VERSION = 0
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "augtoma/medqa_usmle"
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(
                    map(self._process_doc, self.dataset["train"])
                )
            return self._training_docs

    def test_docs(self):
        if self.has_test_docs():
            return map(self._process_doc, self.dataset["test"])


    def _process_doc(self, doc):
        choices = list(doc["options"].values())
        out_doc = {
            "query": doc["question"],
            "choices": choices,
            "gold": ["A", "B", "C", "D"].index(doc["answer_idx"]),
        }
        return out_doc

    def doc_to_text(self, doc):
        return (
                "You are a highly intelligent doctor who answers the following multiple choice question correctly.\nOnly write the answer down."
                + "\n\n**Question:**" + doc["query"] + "\n\n"
                + ",".join(doc["choices"])
                + "\n\n**Answer:**"
        )

    def doc_to_target(self, doc):
        return " " + doc["choices"][doc["gold"]] + "\n\n"
