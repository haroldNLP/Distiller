from Distiller.glue_preprocess import load_and_cache_examples, Processor
from Distiller.glue_preprocess import MrpcProcessor, ColaProcessor, MnliProcessor, MnliMismatchedProcessor, Sst2Processor
from Distiller.glue_preprocess import StsbProcessor, QqpProcessor, QnliProcessor, RteProcessor, WnliProcessor
from Distiller.glue_preprocess import convert_examples_to_features, convert_features_to_dataset
import os
import torch
import argparse
import pandas as pd
from Distiller.transformers import AutoConfig, AutoTokenizer
from Distiller.transformers import AutoModelForSequenceClassification, AutoModelForQuestionAnswering
from torch.utils.data import SequentialSampler, DataLoader

task_dict =  {"mnli" : {0:"entailment", 1:"neutral", 2:"contradiction"},
    "mnli-mm" : {0:"entailment", 1:"neutral", 2:"contradiction"},
    "rte": {0:"entailment", 1:"not_entailment"},
    "qnli":{0:"entailment", 1:"not_entailment"},
    "mrpc":{0:"0", 1:"1"},
    "sst-2":{0:"0", 1:"1"},
    "cola": {0:"0", 1:"1"},
    "qqp": {0:"0", 1:"1"}}


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
}

def main(args):
    config = AutoConfig.from_pretrained(args.model_path)
    args.model_type = config.model_type
    ## load pretrained models and tokenizers
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except Exception as e:
        print(e)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, config=config)

    model.to('cuda')
    processor = glue_processors[args.task_name]()
    examples = processor.get_test_examples(args.dataset_path)
    features = convert_examples_to_features(examples, tokenizer, task=args.task_name, max_length=args.max_seq_length,
                                            label_list=processor.get_labels(),
                                            output_mode=glue_output_modes[args.task_name])
    dataset = convert_features_to_dataset(features, is_testing=True)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=32)
    # if args.task_name is not None:
    #     metric = load_metric("glue", args.task_name)
    preds = []
    label_list = []
    model.eval()
    for step, batch in enumerate(eval_dataloader):

        # labels = batch['labels']
        # batch = tuple(t.to(args.device) for t in batch)
        batch = {key: value.to('cuda') for key, value in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        # outputs = model(**batch)
        predictions = outputs.logits.detach().cpu()
        if args.task_name != "stsb":
            predictions = predictions.argmax(dim=-1)
            preds.extend([task_dict[args.task_name][int(i)] for i in predictions])
        else:
            predictions = predictions[:, 0]
            preds.extend(predictions.tolist())
        label_list.extend(batch['labels'].cpu().tolist())
    pd.DataFrame({'index':list(range(len(preds))), 'prediction':preds}).to_csv(args.output_path+'.tsv', sep="\t", index=False)
    if args.task_name == "mnli":
        args.task_name = "mnli-mm"
        processor = glue_processors[args.task_name]()
        examples = processor.get_test_examples(args.dataset_path)
        features = convert_examples_to_features(examples, tokenizer, task=args.task_name,
                                                max_length=args.max_seq_length,
                                                label_list=processor.get_labels(),
                                                output_mode=glue_output_modes[args.task_name])
        dataset = convert_features_to_dataset(features, is_testing=True)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=32)
        # if args.task_name is not None:
        #     metric = load_metric("glue", args.task_name)
        preds = []
        label_list = []
        model.eval()
        for step, batch in enumerate(eval_dataloader):

            # labels = batch['labels']
            # batch = tuple(t.to(args.device) for t in batch)
            batch = {key: value.to('cuda') for key, value in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            # outputs = model(**batch)
            predictions = outputs.logits.detach().cpu()
            if args.task_name != "stsb":
                predictions = predictions.argmax(dim=-1)
                preds.extend([task_dict[args.task_name][int(i)] for i in predictions])
            else:
                predictions = predictions[:, 0]
                preds.extend(predictions.tolist())
            label_list.extend(batch['labels'].cpu().tolist())
        pd.DataFrame({'index':list(range(len(preds))), 'prediction': preds}).to_csv(args.output_path+"m.tsv", sep="\t", index=False)

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
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--task_name", type=str, default="cola",
                        choices=["cola", "sst-2", "mrpc", "stsb", "qqp", "mnli", "mnli-mm", "qnli", "rte", "wnli"],
                        help="Only used when task type is glue")
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--tokenizer_path", default="huawei-noah/TinyBERT_General_4L_312D")
    parser.add_argument("--output_path", default="./predictions/")

    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if args.task_name == 'cola':
        file_name = 'CoLA'
    elif args.task_name == 'mnli':
        file_name = 'MNLI-m'
    elif args.task_name == 'stsb':
        file_name = 'STS-B'
    else:
        file_name = args.task_name.upper()
    args.output_path = args.output_path + '/' + file_name
    main(args)
