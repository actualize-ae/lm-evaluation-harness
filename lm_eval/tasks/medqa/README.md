# MedQA

### Paper

Title: `MedQA: A Large-scale Open Domain Question Answering Dataset from Medical Exams`

Abstract: `https://arxiv.org/abs/2009.13081`

`Open domain question answering (OpenQA) tasks have been recently attracting
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
community in the future.`

Homepage: `https://github.com/jind11/MedQA`


### Citation

```
@article{jin2020disease,
  title={What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams},
  author={Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung and Fang, Hanyi and Szolovits, Peter},
  journal={arXiv preprint arXiv:2009.13081},
  year={2020}
}
```

### Groups and Tasks

#### Groups

* `group_name`: `Short description`

#### Tasks

* `MedQA`: `A Large-scale Open Domain Question Answering Dataset from Medical Exams`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
