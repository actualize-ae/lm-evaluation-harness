# MedMCQA

### Paper

Title: `MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering`

Abstract: `https://arxiv.org/pdf/2203.14371.pdf`

`This paper introduces MedMCQA, a new large-scale, Multiple-Choice Question Answering (MCQA) dataset
designed to address real-world medical entrance exam questions. More than 194k high-quality AIIMS
& NEET PG entrance exam MCQs covering 2.4k healthcare topics and 21 medical subjects are collected
with an average token length of 12.77 and high topical diversity. Each sample contains a question,
correct answer(s), and other options which requires a deeper language understanding as it tests the
10+ reasoning abilities of a model across a wide range of medical subjects & topics. A detailed explanation
of the solution, along with the above information, is provided in this study.`

Homepage: `https://medmcqa.github.io`


### Citation

```
@InProceedings{pmlr-v174-pal22a,
  title = 	 {MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering},
  author =       {Pal, Ankit and Umapathi, Logesh Kumar and Sankarasubbu, Malaikannan},
  booktitle = 	 {Proceedings of the Conference on Health, Inference, and Learning},
  pages = 	 {248--260},
  year = 	 {2022},
  editor = 	 {Flores, Gerardo and Chen, George H and Pollard, Tom and Ho, Joyce C and Naumann, Tristan},
  volume = 	 {174},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {07--08 Apr},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v174/pal22a/pal22a.pdf},
  url = 	 {https://proceedings.mlr.press/v174/pal22a.html},
  abstract = 	 {This paper introduces MedMCQA, a new large-scale, Multiple-Choice Question Answering (MCQA) dataset designed to address real-world medical entrance exam questions. More than 194k high-quality AIIMS &amp; NEET PG entrance exam MCQs covering 2.4k healthcare topics and 21 medical subjects are collected with an average token length of 12.77 and high topical diversity. Each sample contains a question, correct answer(s), and other options which requires a deeper language understanding as it tests the 10+ reasoning abilities of a model across a wide range of medical subjects &amp; topics. A detailed explanation of the solution, along with the above information, is provided in this study.}
}
```

### Groups and Tasks

#### Groups

* `group_name`: `Short description`

#### Tasks

* `MedMCQA`: `A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
