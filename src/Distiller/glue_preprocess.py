# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" GLUE processors and helpers """

import os
import csv
import json
import torch
import dataclasses
from dataclasses import asdict, dataclass
from multiprocessing import cpu_count, Pool
from enum import Enum
from tqdm import tqdm
from typing import List, Optional, Union
from .transformers import PreTrainedTokenizer
from .utils import Logger
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, r2_score
from datasets import load_dataset
# logger = logging.getLogger(__name__)
logger = Logger("all.log",level="debug").logger

class MyDataset(Dataset):
    def __init__(self, all_input_ids, all_attention_masks, all_token_type_ids, all_labels, s_all_input_ids=None, s_all_attention_masks=None, s_all_token_type_ids=None):
        super(MyDataset, self).__init__()
        self.all_input_ids = all_input_ids
        self.all_attention_masks = all_attention_masks
        self.all_token_type_ids = all_token_type_ids
        self.all_labels = all_labels
        self.s_all_input_ids = s_all_input_ids
        self.s_all_attention_masks = s_all_attention_masks
        self.s_all_token_type_ids = s_all_token_type_ids

    def __getitem__(self, index):
        input_ids = self.all_input_ids[index]
        attention_masks = self.all_attention_masks[index]
        token_type_ids = self.all_token_type_ids[index]
        labels = self.all_labels[index]
        if self.s_all_input_ids is None:
            return {'input_ids': input_ids,
                    'attention_mask': attention_masks,
                    'token_type_ids': token_type_ids,
                    'labels': labels}
        else:
            s_input_ids = self.s_all_input_ids[index]
            s_attention_masks = self.s_all_attention_masks[index]
            s_token_type_ids = self.s_all_token_type_ids[index]
            return {'teacher':{'input_ids': input_ids,
                    'attention_mask': attention_masks,
                    'token_type_ids': token_type_ids,
                    'labels': labels},"student":{'input_ids': s_input_ids,
                    'attention_mask': s_attention_masks,
                    'token_type_ids': s_token_type_ids,
                    'labels': labels}}

    def __len__(self):
        return len(self.all_input_ids)


class ExampleDataset(Dataset):
    def __init__(self, examples):
        super(ExampleDataset, self).__init__()
        self.examples = examples

    def __getitem__(self, index):
        return self.examples[index]



    def __len__(self):
        return len(self.examples)


@dataclass
class InputExample:
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data. Property names are the same names as the corresponding inputs to a model.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``: Usually ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded)
            tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"

class DataProcessor:
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """
        Gets an example from a dict with tensorflow tensors.
        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def tfds_map(self, example):
        """
        Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are. This method converts
        examples to the correct format.
        """
        if len(self.get_labels()) > 1:
            example.label = self.get_labels()[int(example.label)]
        return example

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))


class Processor:
    def __init__(self, args, tokenizer, task, max_length, s_tokenizer=None):
        self.args = args
        self.tokenizer = tokenizer
        self.task = task
        self.max_length = max_length
        self.s_tokenizer = s_tokenizer
    def load_and_cache_examples(self, mode, return_examples=False):
        s_dataset = None
        s_features = None
        s_cached_features_file = None
        # if self.args.local_rank not in [-1, 0] and mode != "dev":
        #     torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Load data features from cache or dataset file
        cached_features_file = os.path.join(self.args.data_dir, "cached_{}_{}_{}_{}".format(mode, self.args.task_name,
                                                                                       list(filter(None,
                                                                                                   self.tokenizer.name_or_path.split(
                                                                                                       "/"))).pop(),
                                                                                       str(self.args.max_seq_length)))
        if self.s_tokenizer:
            s_cached_features_file = os.path.join(self.args.data_dir, "cached_{}_{}_{}_{}".format(mode, self.args.task_name,
                                                                                             list(filter(None,
                                                                                                         self.s_tokenizer.name_or_path.split(
                                                                                                             "/"))).pop(),
                                                                                             str(self.args.max_seq_length)))

        processor = glue_processors[self.args.task_name]()
        examples = processor.get_dev_examples(self.args.data_dir) if mode == "dev" else processor.get_train_examples(
            self.args.data_dir)
        if self.args.repeated_aug and mode == 'train':
            return convert_examples_to_dataset(examples), None, None, None, examples
        if os.path.exists(cached_features_file) and not self.args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
            ## This place need to be more flexible
        else:
            logger.info("Creating features from dataset file at %s", self.args.data_dir)
            features = self.convert_examples_to_features(examples, self.tokenizer, task=self.args.task_name,
                                                    max_length=self.args.max_seq_length,
                                                    label_list=processor.get_labels(),
                                                    output_mode=glue_output_modes[self.args.task_name])
            if self.args.local_rank in [-1, 0]:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)
        # dataset = convert_features_to_dataset(features, is_training=(mode == 'train'))
        if self.s_tokenizer:
            if os.path.exists(s_cached_features_file) and not self.args.overwrite_cache:
                logger.info("Loading student features from cached file %s", s_cached_features_file)
                s_features = torch.load(s_cached_features_file)
            else:
                logger.info("Creating student features from dataset file at %s", self.args.data_dir)
                s_features = self.convert_examples_to_features(examples, self.s_tokenizer, task=self.args.task_name,
                                                          max_length=self.args.max_seq_length,
                                                          label_list=processor.get_labels(),
                                                          output_mode=glue_output_modes[self.args.task_name])
                if self.args.local_rank in [-1, 0]:
                    logger.info("Saving student features into cached file %s", s_cached_features_file)
                    torch.save(s_features, s_cached_features_file)
            # s_dataset = convert_features_to_dataset(s_features, is_training=(mode == 'train'))
        dataset = self.convert_features_to_dataset(features, s_features, is_training=(mode == 'train'))
        # torch.save(dataset,'dataset.bin')
        # torch.save(s_dataset, 's_dataset.bin')
        # torch.save(features, 'features.bin')
        # torch.save(s_features, 's_features.bin')
        # torch.save(examples, 'examples.bin')
        # if self.args.local_rank == 0 and mode != "dev":
        #     torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        # if self.args.local_rank not in [-1,0] and mode != "dev":
        #     dataset = torch.load('dataset.bin')
        #     s_dataset = torch.load('s_dataset.bin')
        #     features = torch.load('features.bin')
        #     s_features = torch.load('s_features.bin')
        #     examples = torch.load('examples.bin')
        if return_examples:
            return dataset, s_dataset, features, s_features, examples
        return dataset

    def convert_examples_to_dataset(self, examples):
        return ExampleDataset(examples)

    def convert_features_to_bacth(self, features, s_features=None):
        s_all_input_ids = None
        s_all_attention_masks = None
        s_all_token_type_ids = None
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        if s_features:
            s_all_input_ids = torch.tensor([f.input_ids for f in s_features], dtype=torch.long)
            s_all_attention_masks = torch.tensor([f.attention_mask for f in s_features], dtype=torch.long)
            s_all_token_type_ids = torch.tensor([f.token_type_ids for f in s_features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        if s_features:
            return {'teacher': {'input_ids': all_input_ids,
                                'attention_mask': all_attention_masks,
                                'token_type_ids': all_token_type_ids,
                                'labels': all_labels}, "student": {'input_ids': s_all_input_ids,
                                                               'attention_mask': s_all_attention_masks,
                                                               'token_type_ids': s_all_token_type_ids,
                                                               'labels': all_labels}}
        else:
            return {'input_ids': all_input_ids,
                    'attention_mask': all_attention_masks,
                    'token_type_ids': all_token_type_ids,
                    'labels': all_labels}


    def convert_features_to_dataset(self,features, s_features=None, is_training=True):
        # Convert to Tensors and build dataset
        s_all_input_ids = None
        s_all_attention_masks = None
        s_all_token_type_ids = None
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        if s_features:
            s_all_input_ids = torch.tensor([f.input_ids for f in s_features], dtype=torch.long)
            s_all_attention_masks = torch.tensor([f.attention_mask for f in s_features], dtype=torch.long)
            s_all_token_type_ids = torch.tensor([f.token_type_ids for f in s_features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        return MyDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_labels, s_all_input_ids,
                         s_all_attention_masks, s_all_token_type_ids)
        # if is_training:
        #     return MyDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_labels)
        # else:
        #     return TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_labels)

    def convert_examples_to_features(
            self,
            examples: Union[List[InputExample], "tf.data.Dataset"],
            label_list=None,
            output_mode=None,
            disable=False,
    ):
        """
        Loads a data file into a list of ``InputFeatures``
        Args:
            examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
            tokenizer: Instance of a tokenizer that will tokenize the examples
            max_length: Maximum example length. Defaults to the tokenizer's max_len
            task: GLUE task
            label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
            output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        Returns:
            If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset`` containing the
            task-specific features. If the input is a list of ``InputExamples``, will return a list of task-specific
            ``InputFeatures`` which can be fed to the model.
        """
        return self._glue_convert_examples_to_features(
            examples, self.tokenizer, label_list=label_list, output_mode=output_mode, disable=disable
        ), self._glue_convert_examples_to_features(
            examples, self.s_tokenizer, label_list=label_list, output_mode=output_mode, disable=disable
        ) if self.s_tokenizer else None

    def _glue_convert_examples_to_features(
            self,
            examples: List[InputExample],
            tokenizer: PreTrainedTokenizer,
            label_list=None,
            output_mode=None,
            disable=False,
    ):
        if self.max_length is None:
            max_length = tokenizer.model_max_length

        if self.task is not None:
            processor = glue_processors[self.task]()
            if label_list is None:
                label_list = processor.get_labels()
                # logger.info(f"Using label list {label_list} for task {self.task}")
            if output_mode is None:
                output_mode = glue_output_modes[self.task]
                # logger.info(f"Using output mode {output_mode} for task {self.task}")

        label_map = {label: i for i, label in enumerate(label_list)}

        def label_from_example(example: InputExample) -> Union[int, float, None]:
            if example.label is None:
                return None
            if output_mode == "classification":
                return label_map[example.label]
            elif output_mode == "regression":
                return float(example.label)
            raise KeyError(output_mode)

        labels = [label_from_example(example) for example in examples]

        batch_encoding = tokenizer(
            [(example.text_a, example.text_b) if example.text_b else example.text_a for example in examples],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True
        )

        features = []
        for i in tqdm(range(len(examples)),disable=disable):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}

            feature = InputFeatures(**inputs, label=labels[i])
            features.append(feature)

        # for i, example in enumerate(examples[:5]):
        #     logger.info("*** Example ***")
        #     logger.info(f"guid: {example.guid}")
        #     logger.info(f"features: {features[i]}")

        return features


def load_and_cache_examples(args, tokenizer, mode, return_examples=False, s_tokenizer=None):
    s_dataset = None
    s_features = None
    s_cached_features_file = None
    # if args.local_rank not in [-1, 0] and mode == "train":
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}_{}".format(mode, args.task_name,
                                                                                   list(filter(None,
                                                                                               tokenizer.name_or_path.split(
                                                                                                   "/"))).pop(),
                                                                                   str(args.max_seq_length)))
    if s_tokenizer:
        s_cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}_{}".format(mode, args.task_name,
                                                                                         list(filter(None,
                                                                                                     s_tokenizer.name_or_path.split(
                                                                                                         "/"))).pop(),
                                                                                         str(args.max_seq_length)))

    processor = glue_processors[args.task_name]()
    examples = processor.get_dev_examples(args.data_dir) if mode == "dev" else processor.get_train_examples(args.data_dir)
    # if args.repeated_aug>1 and mode == 'train':
    #     return convert_examples_to_dataset(examples), None, None, None, examples
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        ## This place need to be more flexible
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        features = convert_examples_to_features(examples, tokenizer, task=args.task_name, max_length=args.max_seq_length,
                                                label_list=processor.get_labels(), output_mode=glue_output_modes[args.task_name])
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    # dataset = convert_features_to_dataset(features, is_training=(mode == 'train'))
    if s_tokenizer:
        if os.path.exists(s_cached_features_file) and not args.overwrite_cache:
            logger.info("Loading student features from cached file %s", s_cached_features_file)
            s_features = torch.load(s_cached_features_file)
        else:
            logger.info("Creating student features from dataset file at %s", args.data_dir)
            s_features = convert_examples_to_features(examples, s_tokenizer, task=args.task_name, max_length=args.max_seq_length,
                                                    label_list=processor.get_labels(), output_mode=glue_output_modes[args.task_name])
            if args.local_rank in [-1, 0]:
                logger.info("Saving student features into cached file %s", s_cached_features_file)
                torch.save(s_features, s_cached_features_file)
        # s_dataset = convert_features_to_dataset(s_features, is_training=(mode == 'train'))
    dataset = convert_features_to_dataset(features, s_features, is_testing=(mode == 'test'))
    # torch.save(dataset, 'dataset.bin')
    # torch.save(s_dataset, 's_dataset.bin')
    # torch.save(features, 'features.bin')
    # torch.save(s_features, 's_features.bin')
    # torch.save(examples, 'examples.bin')
    # if args.local_rank == 0 and mode == "train":
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # if args.local_rank not in [-1,0] and mode != "dev":
    #     dataset = torch.load('dataset.bin')
    #     s_dataset = torch.load('s_dataset.bin')
    #     features = torch.load('features.bin')
    #     s_features = torch.load('s_features.bin')
    #     examples = torch.load('examples.bin')
    if return_examples:
        return dataset, s_dataset, features, s_features, examples
    return dataset


def convert_examples_to_dataset(examples):
    return ExampleDataset(examples)


def convert_features_to_dataset(features, s_features=None, is_testing=False):
    # Convert to Tensors and build dataset
    s_all_input_ids = None
    s_all_attention_masks = None
    s_all_token_type_ids = None
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if s_features:
        s_all_input_ids = torch.tensor([f.input_ids for f in s_features], dtype=torch.long)
        s_all_attention_masks = torch.tensor([f.attention_mask for f in s_features], dtype=torch.long)
        s_all_token_type_ids = torch.tensor([f.token_type_ids for f in s_features], dtype=torch.long)
    if is_testing:
        all_labels = torch.LongTensor([1] * len(features))
    else:
        all_labels = torch.tensor([f.label for f in features])
    return MyDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_labels, s_all_input_ids, s_all_attention_masks, s_all_token_type_ids)
    # if is_training:
    #     return MyDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_labels)
    # else:
    #     return TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_labels)
def convert_examples_to_features(
    examples: Union[List[InputExample], "tf.data.Dataset"],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset`` containing the
        task-specific features. If the input is a list of ``InputExamples``, will return a list of task-specific
        ``InputFeatures`` which can be fed to the model.
    """
    return _glue_convert_examples_to_features(
        examples, tokenizer, max_length=max_length, task=task, label_list=label_list, output_mode=output_mode
    )


def _glue_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.model_max_length

    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info(f"Using label list {label_list} for task {task}")
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info(f"Using output mode {output_mode} for task {task}")

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) if example.text_b else example.text_a for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_token_type_ids=True
    )

    features = []
    for i in tqdm(range(len(examples))):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    # for i, example in enumerate(examples[:5]):
    #     logger.info("*** Example ***")
    #     logger.info(f"guid: {example.guid}")
    #     logger.info(f"features: {features[i]}")

    return features


class OutputMode(Enum):
    classification = "classification"
    regression = "regression"


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info(f"LOOKING AT {os.path.join(data_dir, 'train.tsv')}")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = line[3]
            text_b = line[4]
            label = None if set_type == "test" else line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "neutral","contradiction"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{line[0]}"
            text_a = line[8]
            text_b = line[9]
            label = None if set_type.startswith("test") else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_mismatched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")), "test_mismatched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        if test_mode:
            lines = lines[1:]
        text_index = 1 if test_mode else 3
        examples = []
        for (i, line) in enumerate(lines):
            guid = f"{set_type}-{i}"
            text_a = line[text_index]
            label = None if test_mode else line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 1 if set_type == "test" else 0
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = line[text_index]
            label = None if set_type == "test" else line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
import pandas as pd

class FakeProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, df, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_a = df["description"].values.tolist()
        ids = df["index"].values.tolist()
        for i in range(df.shape[0]):
            label = None if set_type == "test" else str(df.loc[i, "fraudulent"])
            examples.append(InputExample(guid=f"{set_type}-{ids[i]}", text_a=text_a[i], text_b=None, label=label))
        return examples


class ProdProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _create_examples(self, df, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_a = df["description"].values.tolist()
        ids = df["index"].values.tolist()
        for i in range(df.shape[0]):
            label = None if set_type == "test" else str(df.loc[i, "Sentiment"])
            examples.append(InputExample(guid=f"{set_type}-{ids[i]}", text_a=text_a[i], text_b=None, label=label))
        return examples


class ClothProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, df, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_a = df["description"].values.tolist()
        ids = df["index"].values.tolist()
        for i in range(df.shape[0]):
            label = None if set_type == "test" else str(df.loc[i, "Rating"])
            examples.append(InputExample(guid=f"{set_type}-{ids[i]}", text_a=text_a[i], text_b=None, label=label))
        return examples

class KaggleProcessor(DataProcessor):
    """Processor for the kaggle sentiment classification data set (https://www.kaggle.com/akash14/product-sentiment-classification)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "Train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "Dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "Test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _create_examples(self, df, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        df["Product_Description"] = "Product type is " + df["Product_Type"].astype(str) + "." + df["Product_Description"]
        text_a = df["Product_Description"].values.tolist()
        ids = df["Text_ID"].values.tolist()
        for i in range(df.shape[0]):
            label = None if set_type == "test" else str(df.loc[i, "Sentiment"])
            examples.append(InputExample(guid=f"{set_type}-{ids[i]}", text_a=text_a[i], text_b=None, label=label))
        return examples
class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{line[0]}"
            text_a = line[7]
            text_b = line[8]
            label = None if set_type == "test" else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question1"].numpy().decode("utf-8"),
            tensor_dict["question2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        q1_index = 1 if test_mode else 3
        q2_index = 2 if test_mode else 4
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{line[0]}"
            try:
                text_a = line[q1_index]
                text_b = line[q2_index]
                label = None if test_mode else line[5]
            except IndexError:
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{line[0]}"
            text_a = line[1]
            text_b = line[2]
            label = None if set_type == "test" else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{line[0]}"
            text_a = line[1]
            text_b = line[2]
            label = None if set_type == "test" else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples



class BoolQProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = load_dataset("boolq")
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.dataset['train'], "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.dataset['validation'], "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.dataset['validation'], "test")

    def get_labels(self):
        """See base class."""
        return ["True", "False"]

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(dataset['passage']):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = dataset['question'][i]
            text_b = line
            label = None if set_type == "test" else str(dataset['answer'][i])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{line[0]}"
            text_a = line[1]
            text_b = line[2]
            label = None if set_type == "test" else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


glue_tasks_num_labels = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "stsb": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "kaggle": 4,
    "fake": 2,
    "prod": 4,
    "cloth": 1,
    "boolq": 2
}

glue_processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "stsb": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
    'kaggle': KaggleProcessor,
    "fake": FakeProcessor,
    "prod": ProdProcessor,
    "cloth":ClothProcessor,
    'boolq': BoolQProcessor
}


glue_output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "stsb": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "kaggle": "classification",
    "fake": "classification",
    "prod": "classification",
    "cloth": "regression",
    "boolq": "classification"
}


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def roc_auc(preds, labels):
    return roc_auc_score(labels, preds)


def glue_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "stsb":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"mnli/acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"mnli-mm/acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "hans":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "kaggle":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "fake":
        return {'acc': roc_auc(preds, labels), "roc_auc": roc_auc(preds,labels)}
    elif task_name == "prod":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "cloth":
        return {"acc":r2_score(labels, preds),"r2":r2_score(labels, preds)}
    elif task_name == "boolq":
        return {"acc":simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)