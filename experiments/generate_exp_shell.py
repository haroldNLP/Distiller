import pandas as pd
import os

def change_args(arg, value):
    for i,line in enumerate(lines):
        if line.startswith(f"{arg}="):
            lines[i] = f"{arg}={value}\n"

def add_name(value):
    for i,line in enumerate(lines):
        if line.startswith(f"NAME="):
            lines[i] = line.strip('\n')+"_"+value+"\n"


df = pd.read_csv('../auto_distiller_candidates.csv')
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)
df["distill_result"] = 0
for i in range(df.shape[0]):
    with open('distillation.sh') as f:
        lines = f.readlines()
    print(df.loc[i,:])
    best_evaluation = 0.0
    intermediate_loss_type = df.loc[i,'intermediate_loss_type']
    change_args("intermediate_loss_type",intermediate_loss_type)
    change_args("alpha",df.loc[i,'alpha'])
    change_args("intermediate_strategy",df.loc[i,'intermediate_strategy'])
    change_args("kd_loss_type",df.loc[i,'kd_loss_type'])
    change_args("mixup",(str(df.loc[i, 'mixup']) == "True"))
    change_args("aug_p",float(df.loc[i,'aug_p']))
#     args.intermediate_loss_type = df.loc[i,'intermediate_loss_type']
#     args.alpha = df.loc[i, 'alpha']
#     args.intermediate_strategy = df.loc[i, 'intermediate_strategy']
#     args.kd_loss_type = df.loc[i, 'kd_loss_type']
#     args.mixup = (str(df.loc[i, 'kd_loss_type']) == "TRUE")
#     args.aug_p = float(df.loc[i,'aug_p'])
    contextual = int(df.loc[i, 'contextual'])
    backtranslation = int(df.loc[i, 'backtranslation'])
    random_aug = int(df.loc[i, 'random'])

    if contextual != 0 or backtranslation != 0 or random_aug != 0:
#         args.aug_pipeline = True
        change_args("aug_pipeline",True)
        w = list(range(max(contextual, random_aug, backtranslation)))
        if contextual != 0:
            w[contextual-1] = 0
        if backtranslation != 0:
            w[backtranslation-1] = 1
        if random_aug != 0:
            w[random_aug-1] = 2
        change_args("w",'"'+' '.join([str(i) for i in w])+'"')
        add_name('_'.join([str(i) for i in w]))
    else:
        lines.pop(-9)
    with open(f'./tmp/random_distillation_{i}_cloth.sh','w') as f:
        f.writelines(lines)