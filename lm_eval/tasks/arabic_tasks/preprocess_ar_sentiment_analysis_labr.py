import datasets


def process_docs(dataset: datasets.Dataset):

    choices =['1','2','3','4', '5']


    def _helper(doc):
      # modifies the contents of a single
      # document in our dataset.
      doc["query"]= doc["text"]  # The query prompt.
      doc["choices"] = choices
      doc["gold"] = doc["label"]
      return doc

    return dataset.map(_helper) # returns back a datasets.Dataset object

def doc_to_text(doc) -> str:
    return (
            "You are a highly intelligent Arabic speaker  who analyze the following texts and answers with the sentiment analysis.\nOnly write the answer down."
            + "\n\n**Text:**" + doc["query"] + "\n\n"
            + ",".join(doc['choices'])
            + "\n\n**Answer:**"
    )


def doc_to_target(doc) -> int:
    return " " + doc["choices"][doc["gold"]] + "\n\n"
