"""
What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams
https://arxiv.org/abs/2009.13081
This contains the English portion of the full MedQA dataset, containing 12,723 multiple (4) choice questions from the US medical licensing exam.
Homepage: https://paperswithcode.com/dataset/medqa-usmle
"""
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@misc{jin2020disease,
    title={What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams}, 
    author={Di Jin and Eileen Pan and Nassim Oufattole and Wei-Hung Weng and Hanyi Fang and Peter Szolovits},
    year={2020},
    eprint={2009.13081},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""


class MedQA_USMLE_STEP_3(MultipleChoiceTask):
    VERSION = 0
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "augtoma/usmle_step_3"
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        if self.has_test_docs():
            return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        choices = [choice for choice in doc["options"].values() if choice is not None]
        out_doc = {
            "query": doc["question"],
            "choices": choices,
            "gold": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'].index(doc['answer_idx']),
        }
        return out_doc

    def doc_to_text(self, doc):
        return (
                "You are a highly intelligent doctor who answers the following multiple choice question correctly.\nOnly write the answer down."
                + "\n\n**Question:**" + doc["query"] + "\n\n"
                + ",".join(choice for choice in doc['choices'] if choice is not None)
                + "\n\n**Answer:**"
        )

    def doc_to_target(self, doc):
        return " " + doc["choices"][doc["gold"]] + "\n\n"
