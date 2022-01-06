from tqdm import tqdm
import os
import torch
from preprocessing import SquadResult
from torch.utils.data import DataLoader, SequentialSampler
import argparse
from utils import load_and_cache_examples
import logging

logger = logging.getLogger(__name__ )
def evaluate(args, model, tokenizer, prefix=""):
    dataset, features, examples = load_and_cache_examples(args, tokenizer, mode="dev", return_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

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
    return examples, features, all_results



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, required=True, help="Directory of checkpoints")
