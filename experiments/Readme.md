# How to launch experiments on knowledge distillation

First you shoud prepare & download your data

Use this script to download GLUE datasets to [directory datasets](../datasets)

```shell
python3 ../download_glue_data.py --data_dir ../datasets/glue_data --tasks all
```

And you can download SQuAD data [here](https://rajpurkar.github.io/SQuAD-explorer/)

And then you need to install the dependencies by running:

```shell
pip3 install -r ../src/Distiller/requirements.txt
```

Remember to install torch which is compatible with your CUDA version!

Then you can launch the experiment by:

```shell
bash distillation.sh
```

Or you can execute as a python script by:

```python
python ../src/Distiller/distiller.py -- \
    --task_type glue \
    --task_name sst-2 \
    --data_dir ../datasets/glue_data \
    --T_model_name_or_path howey/electra-base-sst2 \
    --S_model_name_or_path howey/electra-small-sst2 \
    --output_dir output-student \
    --max_seq_length 128 \
    --train \
    --eval \
    --doc_stride 128 \
    --per_gpu_train_batch_size 32 \
    --seed 40 \
    --num_train_epochs 20 \
    --learning_rate 1e-4 \
    --thread 64 \
    --gradient_accumulation_steps 1} \
    --temperature 8 \
    --kd_loss_weight 1.0 \
    --kd_loss_type ce
```

For the meaning of hyperparameters, see [configs.py](../src/Distiller/configs.py)

At this time, we can run experiments on SQuAD and GLUE, if you want to try experiments on other baselines, first rewrite a preprocess function such as [glue_preprocess](../src/Distiller/glue_preprocess.py) , and then import them in [distiller.py](../src/Distiller/distiller.py)

