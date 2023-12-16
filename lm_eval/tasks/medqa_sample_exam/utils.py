def process_docs(dataset):
    def _process(doc):
        choices = [choice for choice in doc["options"].values() if choice is not None]
        out_doc = {
            "query": doc["question"],
            "choices": choices,
            "gold": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'].index(doc['answer_idx']),
        }
        return out_doc

    return dataset.map(_process)


def doc_to_text(self, doc):
    return (
            "You are a highly intelligent doctor who answers the following multiple choice question correctly.\nOnly write the answer down."
            + "\n\n**Question:**" + doc["query"] + "\n\n"
            + ",".join(choice for choice in doc['choices'] if choice is not None)
            + "\n\n**Answer:**"
    )


def doc_to_target(self, doc):
    return " " + doc["choices"][doc["gold"]] + "\n\n"
