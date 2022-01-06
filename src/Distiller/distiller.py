import os
import json
import glob
import torch
import logging
import random
from ray import tune
import numpy as np
from .configs import parse
from .autoaug import AutoAugmenter
from .utils import Logger, cal_layer_mapping
from .mp_aug import aug_process
from .transformers import AutoConfig, AutoTokenizer
from .transformers import AutoModelForSequenceClassification, AutoModelForQuestionAnswering
from .textbrewer import DistillationConfig,TrainingConfig,GeneralDistiller, EMDDistiller
import queue
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from .transformers import AdamW, get_linear_schedule_with_warmup, WEIGHTS_NAME
from torch.multiprocessing import Queue, Process, set_start_method

logger = logging.getLogger(__name__)
task_dict = {'squad2': AutoModelForQuestionAnswering,
             'squad': AutoModelForQuestionAnswering,
             'glue': AutoModelForSequenceClassification,
             'superglue': AutoModelForSequenceClassification}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


# class CustomDataLoader(DataLoader):


def train(args, examples, train_dataset, t_model, s_model, tokenizer, augmenter=None, matches=None, predict_callback=None, q=None):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    # mix_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    # train_dataloader = CustomDataLoader(train_dataset, examples, args=args, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate_fn, tokenizer=tokenizer, augmenter=augmenter)
    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate_fn)
    # def collate_fn(batch):
    #     return [({i:k[0] for i,k in piece.items()}) for piece in batch], [({i:k[0] for i,k in piece.items()}) for piece in batch]
    if augmenter:
        QUEUE_LIMIT = 60
        count = 0
        while count<QUEUE_LIMIT:
            try:
                count+=1
                train_dataset = q.get(timeout=60)
                break
            except queue.Empty:
                logger.info("Waiting for data augmentation process to return data")
        # train_dataloader = DataProvider(train_dataset, examples, args, tokenizer, augmenter,s_tokenizer,s_dataset)

    # else:
    #     train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    #     train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    # mix_dataloader = DataLoader(train_dataset, sampler=mix_sampler,
    #                             batch_size=args.train_batch_size) if args.mixup else None
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        # t_total =
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in s_model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in s_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler_class = get_linear_schedule_with_warmup
    args.warmup_steps = int(t_total * args.warmup_proportion)
    scheduler_args = {'num_warmup_steps': args.warmup_steps, 'num_training_steps': t_total}
    # if args.fp16:
    #     try:
    #         from apex import amp
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #     s_model, optimizer = amp.initialize(s_model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1 and args.local_rank == -1:
        t_model = torch.nn.DataParallel(t_model)
        s_model = torch.nn.DataParallel(s_model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        s_model = torch.nn.parallel.DistributedDataParallel(s_model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
        t_model = torch.nn.parallel.DistributedDataParallel(t_model, device_ids=[args.local_rank],
                                                            output_device=args.local_rank,
                                                            find_unused_parameters=True)
    actual_batch_size = args.per_gpu_train_batch_size
    num_train_steps = len(train_dataloader) // args.gradient_accumulation_steps * actual_batch_size
    if augmenter:
        actual_batch_size *= 2
    if args.mixup:
        actual_batch_size *= 2

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", actual_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                actual_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Actual train batch size (w. mixup & data augmentation) = %d", actual_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    if args.train:
        if args.intermediate_strategy and args.intermediate_strategy.lower() == "emd":
            distill_config = DistillationConfig(
                temperature=args.temperature,
                kd_loss_weight=args.kd_loss_weight,
                kd_loss_type=args.kd_loss_type)
        else:
            # intermediate_matches = matches
            # if args.intermediate_strategy == "skip":
            #     intermediate_matches = []
            #     for match in matches:
            #         intermediate_matches += matches[match]
            logger.info(f"{matches}")
            distill_config = DistillationConfig(
                temperature=args.temperature,
                intermediate_matches=matches,
                kd_loss_weight=args.kd_loss_weight,
                kd_loss_type=args.kd_loss_type)
        train_config = TrainingConfig(gradient_accumulation_steps=args.gradient_accumulation_steps, device=args.device,
                                      log_dir=os.path.join(args.output_dir, "log"), output_dir=args.output_dir,
                                      fp16=args.fp16, mixup=args.mixup, local_rank=args.local_rank,
                                      task_type=args.task_type, q=q)
        if args.task_type in ["squad", "squad2"]:
            args.task_name = args.task_type
            from .adapters import BertForQAAdaptor as adaptor_func
        elif args.task_type == "glue":
            from .adapters import BertForGLUEAdptor as adaptor_func
        adaptor_T = adaptor_func
        adaptor_S = adaptor_func
        if args.intermediate_strategy == "EMD":
            distiller = EMDDistiller(train_config=train_config,
                                     distill_config=distill_config,
                                     model_T=t_model, model_S=s_model,
                                     adaptor_T=adaptor_T,
                                     adaptor_S=adaptor_S,
                                     emd=matches)
        else:
            distiller = GeneralDistiller(train_config, distill_config, t_model, s_model, adaptor_T, adaptor_S, )
        with distiller:
            distiller.train(optimizer, scheduler_class=scheduler_class, scheduler_args=scheduler_args, dataloader=train_dataloader,
                            num_epochs=args.num_train_epochs, callback=predict_callback,max_grad_norm=args.max_grad_norm)
            # distiller.train(optimizer,train_dataloader,args.num_train_epochs,
            #                 scheduler_class=scheduler_class, scheduler_args=scheduler_args,
            #                 max_grad_norm=1.0, callback=predict_callback, mixup_value=args.mixup_value,
            #                 mix_dataloader=mix_dataloader, local_rank=args.local_rank)
    return


def main(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)
    ## load pretrained models and tokenizers
    if args.local_rank not in [-1,0]:
        torch.distributed.barrier()

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
                                                config=t_config)
    s_tokenizer = AutoTokenizer.from_pretrained(args.S_model_name_or_path,
                                                use_fast=False,
                                                config=s_config) if args.S_model_name_or_path != args.T_model_name_or_path else None
    ## Initialize augmenter

    model_class = task_dict.get(args.task_type)
    t_model = model_class.from_pretrained(args.T_model_name_or_path, config=t_config)
    ## If the student borrow layers from teachers, it must borrow complete layers. Their hidden size and attention size
    # must be the same
    if args.random_student:
        s_model = model_class.from_config(s_config)
    else:
        s_model = model_class.from_pretrained(args.S_model_name_or_path, config=s_config)
    if args.local_rank == 0:
        torch.distributed.barrier()

    logger.info("Training/evaluation parameters %s", args)
    s_model.to(args.device)
    t_model.to(args.device)
    def predict_callback(model, step):
        if args.eval and args.local_rank in [-1, 0]:
            evaluation_result = evaluate_func(args, model, s_tokenizer if s_tokenizer else t_tokenizer, prefix=step)
            logger.info("***** Eval results *****")
            logger.info(json.dumps(evaluation_result, indent=2) + '\n')

            output_eval_file = os.path.join(args.output_dir, f"{step}_eval_results.txt")
            logger.info(f"Write evaluation result to {output_eval_file}...")
            with open(output_eval_file, "a") as writer:
                writer.write(f"Output: {json.dumps(evaluation_result, indent=2)}\n")
            # if "exact" in evaluation_result.keys():
            #     return evaluation_result['exact'], evaluation_result['f1']
            # else:
            #     return evaluation_result
            model.train()
            try:
                tune.report(iterations=step, accuracy=evaluation_result['acc'])
            except Exception as e:
                logger.warning(e)
            return list(evaluation_result.values())[0]
        else:
            return None
    ## Training
    if args.train:
        # examples = read_examples_from_file(args.data_dir, mode="train", task_type=args.task_type)
        matches = cal_layer_mapping(args, t_config, s_config)
        train_dataset, s_dataset, features, s_features, examples = load_and_cache_examples(args, t_tokenizer, mode="train",
                                                                return_examples=True, s_tokenizer=s_tokenizer)
        augmenter = None
        q = None
        # if args.augmenter_config_path:
        #     augmenter = AutoAugmenter.from_config(args.augmenter_config_path, "cpu" if args.n_gpu == 0 else "gpu")
        #     # global q
        #     q = Queue()
        #     process = Process(target=aug_process,
        #                       args=(q, examples, train_dataset, augmenter, args, t_tokenizer, s_tokenizer))
        #     process.start()
        if args.aug_type:
            augmenter = AutoAugmenter.from_config(args.aug_type)
            q = Queue()
            process = Process(target=aug_process,
                              args=(q, examples, train_dataset, augmenter, args, t_tokenizer, s_tokenizer))
            process.start()
            # process.join()
        if args.aug_pipeline:
            augmenter = AutoAugmenter.init_pipeline(w=[1,0,1])
            if len(augmenter):
                args.augs = augmenter.aug_names
                q = Queue()
                process = Process(target=aug_process,
                                  args=(q, examples, train_dataset, augmenter, args, t_tokenizer, s_tokenizer))
                process.start()

        train(args, examples, train_dataset, t_model, s_model, t_tokenizer, augmenter, matches, predict_callback, q=q)
        # p = Process(target=data_aug_process, args=(augmenter,examples,tokenizer,args))
        # p.start()
        # if args.S_model_name_or_path != args.T_model_name_or_path:
        #     s_train_dataset = load_and_cache_examples(args, s_tokenizer, mode="train", model_name_or_path=args.S_model_name_or_path, examples=examples)
        # train(args, examples, train_dataset, t_model, s_model, t_tokenizer, augmenter, matches, predict_callback,s_tokenizer, s_dataset)

# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = s_model.module if hasattr(s_model, "module") else s_model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        if s_tokenizer:
            s_tokenizer.save_pretrained(args.output_dir)
        else:
            t_tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        model = model_class.from_pretrained(args.output_dir)  # , force_download=True)

        # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
        # So we use use_fast=False here for now until Fast-tokenizer-compatible-examples are out

        model.to(args.device)
        # Good practice: save your training arguments together with the trained model

    # Evaluation
    results = {}
    if args.eval and args.local_rank in [-1, 0]:
        if args.train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )

        else:
            logger.info("Loading checkpoint %s for evaluation", args.S_model_name_or_path)
            checkpoints = [args.S_model_name_or_path]

        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            evaluation_result = evaluate_func(args, model, s_tokenizer if s_tokenizer else t_tokenizer, prefix=global_step,write_prediction=True)
            logger.info("***** Eval results *****")
            logger.info(json.dumps(evaluation_result, indent=2) + '\n')

            output_eval_file = os.path.join(args.output_dir, "final_eval_results.txt")
            logger.info(f"Write evaluation result to {output_eval_file}...")
            with open(output_eval_file, "a") as writer:
                writer.write(f"Output: {json.dumps(evaluation_result, indent=2)}\n")
    return


if __name__ == '__main__':
    import redis
    r = redis.Redis()
    args = parse()
    set_start_method('spawn')
    if args.S_model_name_or_path is None:
        args.S_model_name_or_path = args.T_model_name_or_path
    if args.task_type in ["squad","squad2"]:
        args.task_name = args.task_type
        from evaluate import evaluate_squad as evaluate_func
        from squad_preprocess import load_and_cache_examples
        from adapters import BertForQAAdaptor as adaptor_func
    elif args.task_type == "glue":
        from evaluate import evaluate_glue as evaluate_func
        from glue_preprocess import load_and_cache_examples
        from adapters import BertForGLUEAdptor as adaptor_func
    logger = Logger(f"{args.output_dir}/all.log", level="debug").logger
    main(args)