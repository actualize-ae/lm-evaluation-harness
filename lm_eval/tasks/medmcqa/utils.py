def process_docs(dataset):
    def _process(doc):
        choices = [doc['opa'], doc['opb'], doc['opc'], doc['opd']]
        out_doc = {
            "query": doc["question"],
            "choices": choices,
            "gold": doc['cop'],
            "subject": doc['subject_name']
        }
        return out_doc

    return dataset.map(_process)


def doc_to_text(doc):
    # Followed the prompt used in openai Medmcqa evaluation https://github.com/openai/evals/blob/main/evals/registry/data/medmcqa/convert.js
    return (
            "You are a highly intelligent doctor who answers the following multiple choice question correctly.\nOnly write the answer down."
            "\n\n**Subject:**" + doc["subject"]
            + "\n\n**Question:**" + doc["query"] + "\n\n"
            + ",".join(doc['choices'])
            + "\n\n**Answer:**"
    )


def doc_to_target(doc):
    return " " + doc["choices"][doc["gold"]] + "\n\n"
