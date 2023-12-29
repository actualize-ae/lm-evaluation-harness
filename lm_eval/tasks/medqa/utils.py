def process_docs(dataset):
    def _process(doc):
        choices = doc["choices"]

        return {
            "query": doc["question"],  # The query prompt.
            "choices": choices,  # The list of choices.
            "gold": doc["choices"].index(doc["answer"][0]),
        }

    return dataset.map(_process)


def doc_to_text(self, doc):
    return (
            "You are a highly intelligent doctor who answers the following multiple choice question correctly.\nOnly write the answer down."
            + "\n\n**Question:**" + doc["query"] + "\n\n"
            + ",".join(doc['choices'])
            + "\n\n**Answer:**"
    )


def doc_to_target(self, doc):
    return " " + doc["choices"][doc["gold"]] + "\n\n"
