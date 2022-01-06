from tqdm import tqdm
import os
import numpy as np
import torch
from .squad_preprocess import SquadResult
from torch.utils.data import DataLoader, SequentialSampler
from .utils import write_predictions_squad, get_raw_scores, apply_no_ans_threshold, make_eval_dict, merge_eval, find_all_best_thresh_v2
import argparse
import logging
from .glue_preprocess import glue_compute_metrics
logger = logging.getLogger(__name__ )


def evaluate_squad(args, model, tokenizer, prefix="",write_prediction=False):
    from .squad_preprocess import load_and_cache_examples
    dataset, s_dataset, features, s_features, examples = load_and_cache_examples(args, tokenizer, mode="dev", return_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu training (should be after apex fp16 initialization)
    # if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
    #     model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    # if args.local_rank != -1:
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
    #     )

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
                del inputs["token_type_ids"]

            feature_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )
            outputs = model(**inputs)
        batch_start_logits = outputs.start_logits.detach().cpu().tolist()
        batch_end_logits = outputs.end_logits.detach().cpu().tolist()
        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            # output = [output[i].detach().cpu().tolist() for output in outputs.to_tuple()]
            start_logits= batch_start_logits[i]
            end_logits = batch_end_logits[i]
            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            # if len(output) >= 5:
            #     start_top_index = output[1]
            #     end_top_index = output[3]
            #     cls_logits = output.cls_logits
            #
            #     result = SquadResult(
            #         unique_id,
            #         start_logits,
            #         end_logits,
            #         start_top_index=start_top_index,
            #         end_top_index=end_top_index,
            #         cls_logits=cls_logits,
            #     )
            #
            # else:
            result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)
    evaluation = _squad_evaluate(args, tokenizer, examples, features, all_results, prefix, write_prediction)
    return evaluation


def _squad_evaluate(args, tokenizer, eval_examples, eval_features, all_results, prefix="", write_prediction=True,
                   no_answer_probs=None, no_answer_probability_threshold=1.0):
    output_prediction_file = os.path.join(args.output_dir, f"{prefix}_predictions.json")
    output_nbest_file = os.path.join(args.output_dir, f"nbest_predictions_{prefix}.json")
    version_2_with_negative = True if args.task_type=="squad2" else False
    if version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None
    all_predictions = write_predictions_squad(tokenizer, eval_examples, eval_features, all_results,
                                               args.n_best_size, args.max_answer_length,
                                               True, output_prediction_file,
                                               output_nbest_file, output_null_log_odds_file,
                                               version_2_with_negative,
                                               args.null_score_diff_threshold, write_prediction=write_prediction)
    qas_id_to_has_answer = {example.qas_id: bool(example.answers) for example in eval_examples}
    has_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if has_answer]
    no_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if not has_answer]

    if no_answer_probs is None:
        no_answer_probs = {k: 0.0 for k in all_predictions}

    exact, f1 = get_raw_scores(eval_examples, all_predictions)

    exact_threshold = apply_no_ans_threshold(
        exact, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold
    )
    f1_threshold = apply_no_ans_threshold(f1, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold)

    evaluation = make_eval_dict(exact_threshold, f1_threshold)

    if has_answer_qids:
        has_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=has_answer_qids)
        merge_eval(evaluation, has_ans_eval, "HasAns")

    if no_answer_qids:
        no_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=no_answer_qids)
        merge_eval(evaluation, no_ans_eval, "NoAns")

    if no_answer_probs:
        find_all_best_thresh_v2(evaluation, all_predictions, exact, f1, no_answer_probs, qas_id_to_has_answer)
    return evaluation


def evaluate_glue(args, model, tokenizer, prefix="",write_prediction=False):
    from .glue_preprocess import load_and_cache_examples
    dataset, s_dataset, features, s_features, examples = load_and_cache_examples(args, tokenizer, mode="dev",
                                                                                 return_examples=True)
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    # if args.task_name is not None:
    #     metric = load_metric("glue", args.task_name)
    preds = []
    label_list = []
    model.eval()
    for step, batch in enumerate(eval_dataloader):

        # labels = batch['labels']
        # batch = tuple(t.to(args.device) for t in batch)
        batch = {key: value.to(args.device) for key, value in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        # outputs = model(**batch)
        predictions = outputs.logits.detach().cpu()
        if args.task_name not in ["stsb","cloth"]:
            predictions = predictions.argmax(dim=-1)
        else:
            predictions = predictions[:,0]
        label_list.extend(batch['labels'].cpu().tolist())
        preds.extend(predictions.tolist())

    # eval_metric_compute = metric.compute()
    eval_metric = glue_compute_metrics(args.task_name,np.array(preds), np.array(label_list))
    if args.task_name == 'mnli':
        # new_args = args.copy()
        args.task_name = 'mnli-mm'
        dataset, s_dataset, features, s_features, examples = load_and_cache_examples(args, tokenizer, mode="dev",
                                                                                     return_examples=True)

        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
        # if args.task_name is not None:
        #     metric = load_metric("glue", args.task_name)
        preds = []
        label_list = []

        for step, batch in enumerate(eval_dataloader):
            # labels = batch['labels']
            # batch = tuple(t.to(args.device) for t in batch)
            batch = {key: value.to(args.device) for key, value in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            # outputs = model(**batch)
            predictions = outputs.logits.detach().cpu().argmax(dim=-1)
            label_list.extend(batch['labels'].cpu().tolist())
            preds.extend(predictions.tolist())

        # eval_metric_compute = metric.compute()
        mm_eval_metric = glue_compute_metrics(args.task_name, np.array(preds), np.array(label_list))
        eval_metric['mnli-mm/acc'] = mm_eval_metric['mnli-mm/acc']
        eval_metric['m_mm_acc'] = (eval_metric['mnli/acc'] + eval_metric['mnli-mm/acc'])/2
        args.task_name='mnli'
    logger.info(f"step {prefix}: {eval_metric}")
    return eval_metric

