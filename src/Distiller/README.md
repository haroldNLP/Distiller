# Meta-algorithm: Distiller

Teacher network architecture

* -RoBERTa-large-
* ELECTRA-base + ELECTRA-large
* -T5-

Student network architecture

* ELECTRA-small
* MobileBERT
* -Other small architectures-

Data augmentation

* Tiny-BERT-like GloVE + BERT-MLM mask reconstruction
* BART / T5-based sequence reconstruction
    * We mask a fraction of input words and try to reconstruct input words with a pretrained seq-to-seq model.
* Word augmenter: Random Word
* Mix-up KD
* -MODALS: https://openreview.net/pdf?id=XjYgR6gbCEc-
* Repeated augmentation: https://openaccess.thecvf.com/content_CVPR_2020/papers/Hoffer_Augment_Your_Batch_Improving_Generalization_Through_Instance_Repetition_CVPR_2020_paper.pdf
* Back translation

Distillation method

* Objective functions (li,j​(⋅,⋅),l^i​(⋅,⋅)s in the equation)
    * Earth Mover’s Distance(EMD)
        * https://www.researchgate.net/publication/220659330_The_Earth_Mover's_Distance_as_a_Metric_for_Image_Retrieval
    * Random Projection + MSE
    * Mutual Information
        * https://arxiv.org/pdf/1904.05835.pdf
    * Cross Entropy
    * Cosine Distance
    * Patient Knowledge Distillation(PKD)
        * Section 3.2 of https://arxiv.org/pdf/1908.09355.pdf

* Layer mapping
    * We investigate three intermediate distillation strategies:
        * Skip: the student learns from every k layers of the teacher
        * Last: the student learns from the last k layers of the teacher
        * EMD: the student learns from every layer of the teacher by a corresponding layer weight based on EMD
    * Last-layer distillation(not perform well)

* Additional distillation token: https://arxiv.org/pdf/2012.12877.pdf#cite.Cubuk2019RandAugmentPA

Multi-teacher distillation

* Distill from multiple teacher networks

