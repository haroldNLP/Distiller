# Towards Automated Distillation: A Systematic Study of Knowledge Distillation in Natural Language Processing

This repository includes training and testing code of the paper "Towards Automated Distillation: A Systematic Study of Knowledge Distillation in Natural Language Processing". You can reproduce our experimental results with code [here](experiments/). 

## Ray Tune

To use [ray](https://docs.ray.io/en/master/index.html) cluster to run the experiments, first

```shell
pip install -e .
```

then go to [ray_directory](ray_directory) 

## Dataset

For QA task, you can download SQuAD dataset (v1, v2) by using shell command:

``` shell
bash download_squad2.sh
```

For downloading other datasets, follow this: https://github.com/dmlc/gluon-nlp/tree/master/scripts/datasets/question_answering

If you want to download all data of NaturalQuestions(42G), you can check out the official statement of NaturalQuestions here: https://ai.google.com/research/NaturalQuestions/download


