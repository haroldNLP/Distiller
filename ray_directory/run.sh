#aws s3 cp --recursive $OUTPUT_DIR s3://haoyu-nlp/experiments/$OUTPUT_DIR
#set hyperparameters
#BERT_DIR=output-bert-base/squad_base_cased_lr3e2_teacher
TEACHER_DIR=csarron/bert-base-uncased-squad-v1
STUDENT_DIR=huawei-noah/TinyBERT_General_4L_312D
DATA_ROOT_DIR=/home/ray/Distillation_QA_benchmark/datasets/glue_data/
OUTPUT_ROOT_DIR=output-student

#STUDENT_CONF_DIR=student_configs/bert_base_cased_L4.json
accu=1
ep=20
lr=5
alpha=0.9
#augmenter_config_path=augmenter_config.json
intermediate_strategy=skip
intermediate_loss_type=mi
intermediate_features=hidden
kd_loss_type=ce
## if you use mixup or augmenter, then the actual batch size will be batch_size * 2
batch_size=8
temperature=1
length=128
torch_seed=9580
hard_label_weight=0.0
kd_loss_weight=1.0
inter_loss_weight=0.0
task_name=boolq
task_type=glue
aug_p=0.3
w=None
aug_pipeline=False
mixup=False
NAME=${TEACHER_DIR}_${STUDENT_DIR}_lr${lr}e-5_e${ep}_${task_type}_${task_name}_${intermediate_strategy}_${intermediate_loss_type}_alpha${alpha}_h${hard_label_weight}_k${kd_loss_weight}_${kd_loss_type}_${mixup}_${aug_pipeline}_${aug_p}
OUTPUT_DIR=${OUTPUT_ROOT_DIR}/${NAME}

gpu_nums=4

#export CUDA_VISIBLE_DEVICES=0
mkdir -p $OUTPUT_DIR
python run.py \
    --task_type ${task_type} \
    --task_name ${task_name} \
    --data_dir $DATA_ROOT_DIR \
    --T_model_name_or_path $TEACHER_DIR \
    --S_model_name_or_path $STUDENT_DIR \
    --output_dir $OUTPUT_DIR \
    --max_seq_length ${length} \
    --intermediate_strategy ${intermediate_strategy} \
    --intermediate_features ${intermediate_features}\
    --intermediate_loss_type ${intermediate_loss_type} \
    --train \
    --eval \
    --fp16 \
    --doc_stride 128 \
    --per_gpu_train_batch_size ${batch_size} \
    --seed ${torch_seed} \
    --num_train_epochs ${ep} \
    --learning_rate ${lr}e-5 \
    --max_grad_norm -1.0 \
    --thread 64 \
    --gradient_accumulation_steps ${accu} \
    --temperature ${temperature} \
    --alpha ${alpha} \
    --mixup ${mixup}\
    --aug_p ${aug_p} \
    --aug_pipeline ${aug_pipeline} \
    --hard_label_weight ${hard_label_weight} \
    --soft_label_weight 0.0 \
    --num_reaug 5 \
    --kd_loss_weight ${kd_loss_weight} \
    --inter_loss_weight ${inter_loss_weight} \
    --kd_loss_type ${kd_loss_type}

#aws s3 cp --recursive $OUTPUT_DIR s3://haoyu-nlp/experiments/$OUTPUT_DIR
