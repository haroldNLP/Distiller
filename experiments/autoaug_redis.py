import redis
from redis import StrictRedis
import torch
import argparse
from torch.multiprocessing import cpu_count, Pool, Queue
from tqdm import tqdm
from torch.utils.data import ConcatDataset
import time
from Distiller.transformers import AutoConfig, AutoTokenizer
from Distiller.autoaug import AutoAugmenter

def example_iter(examples, batch_size):
    i = 0
    while i < len(examples):
        if (i + batch_size) >= len(examples):
            # yield [j.context_text for j in examples[i:]],i
            yield examples[i:]
        else:
            # yield [j.context_text for j in examples[i:i+32]], i
            yield examples[i:i + batch_size]
        i += batch_size


def augment_data(iter_sample, augmenter, task_type):
    result = iter_sample.copy()
    if task_type in ['squad', 'squad2']:
        for ii, dd in enumerate(augmenter.augment([i.context_text for i in iter_sample])):
            result[ii].context_text = dd
    elif task_type == "glue":
        for ii, dd in enumerate(augmenter.augment([i.text_a for i in iter_sample])):
            result[ii].text_a = dd
        if hasattr(iter_sample[0],"text_b") and iter_sample[0].text_b:
            for ii, dd in enumerate(augmenter.augment([i.text_b for i in iter_sample])):
                result[ii].text_b = dd
    return result

from functools import wraps

def generate_aug_data(examples, augmenter, args, tokenizer, s_tokenizer=None):
    if args.task_type == "glue":
        from Distiller.glue_preprocess import convert_features_to_dataset, convert_examples_to_features
    elif args.task_type in ["squad", "squad2"]:
        from Distiller.squad_preprocess import convert_features_to_dataset, convert_examples_to_features
    else:
        raise NotImplementedError
    threads = min(args.thread, cpu_count())
    from functools import partial
    with Pool(threads) as p:
        # global examples
        # examples = self.examples
        annotate_ = partial(
            augment_data,
            augmenter=augmenter,
            task_type=args.task_type
        )
        aug_examples = list(
            tqdm(
                p.map(annotate_, example_iter(examples,args.per_gpu_train_batch_size)),
                total=int(len(examples) / args.per_gpu_train_batch_size) + 1,
                desc="Data augmentation",
                disable=False,
            )
        )
    new_examples = []
    for i in aug_examples:
        new_examples.extend(i)
    del aug_examples
    # features = convert_examples_to_features(new_examples, tokenizer, args.max_seq_length,
    #                                         task=args.task_name
    #                                         )
    # s_features = None
    # if s_tokenizer:
    #     s_features = convert_examples_to_features(new_examples, s_tokenizer,
    #                                               args.max_seq_length,
    #                                               task=args.task_name
    #                                               )
    #
    # dataset = convert_features_to_dataset(features, s_features)
    # new_dataset = ConcatDataset([original_dataset, dataset])
    # return new_dataset
    return new_examples


def main(args,r):
    args.repeated_aug=True
    t_config = AutoConfig.from_pretrained(args.T_config_file if args.T_config_file else args.T_model_name_or_path)
    s_config = AutoConfig.from_pretrained(args.S_config_file if args.S_config_file else args.S_model_name_or_path)
    args.model_type = s_config.model_type
    s_config.num_labels = t_config.num_labels
    t_config.output_hidden_states = True
    t_config.output_attentions = True
    s_config.output_hidden_states = True
    s_config.output_attentions = True
    t_tokenizer = AutoTokenizer.from_pretrained(args.T_model_name_or_path,
                                                use_fast=False,
                                                config=t_config,
                                                )
    s_tokenizer = AutoTokenizer.from_pretrained(args.S_model_name_or_path,
                                                use_fast=False,
                                                config=s_config) if args.S_model_name_or_path != args.T_model_name_or_path else None
    train_dataset, s_dataset, features, s_features, examples = load_and_cache_examples(args, t_tokenizer, mode="train",
                                                                                       return_examples=True,
                                                                                       s_tokenizer=s_tokenizer)
    augmenter = AutoAugmenter.init_pipeline(w=[1,0,1])

    new_examples = generate_aug_data(examples, augmenter, args, t_tokenizer, s_tokenizer)
    for i in new_examples:
        r.rpush('text_a', i.text_a)
        if hasattr(i, 'text_b') and i.text_b:
            r.rpush('text_a', i.text_b)

    r.lpop('text_a')


if __name__ == '__main__':
    r = redis.Redis()
    from Distiller.configs import parse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--task_name")
    # parser.add_argument("--task_type")
    # parser.add_argument("--data_dir")
    # parser.add_argument("--max_seq_length", default=128)
    # parser.add_argument("--T_model_name_or_path")
    # parser.add_argument("--S_model_name_or_path")
    # parser.add_argument("--T_config_file",default=None,type=str)
    # parser.add_argument("--S_config_file", default=None, type=str)
    # parser.add_argument("--local_rank", default=-1, type=int)
    # args = parser.parse_args()
    args = parse()
    args.output_dir = './'
    if args.task_type in ["squad","squad2"]:
        args.task_name = args.task_type
        from Distiller.evaluate import evaluate_squad as evaluate_func
        from Distiller.squad_preprocess import load_and_cache_examples
        from Distiller.adapters import BertForQAAdaptor as adaptor_func
    elif args.task_type == "glue":
        from Distiller.evaluate import evaluate_glue as evaluate_func
        from Distiller.glue_preprocess import load_and_cache_examples, Processor
        from Distiller.adapters import BertForGLUEAdptor as adaptor_func
    main(args)