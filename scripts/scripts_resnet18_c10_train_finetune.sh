mkdir -p ./out/





# # ### Setting 2:
# export lamda_wgt=16.0
# export lamda_relu=65.0
# export w_mask_lr=0.001
# export w_lr=0.001
# export epochs=250
# export gpu=2

export lamda_wgt=13.0
export lamda_relu=65.0
export w_mask_lr=0.001
export w_lr=0.001
export epochs=250
export gpu=1


#### Additional parameters:
# export w_decay=0.0004 ##### Original network w_decay=0.0001
export finetune_epoch=100
export w_lr_finetune=0.01

#  --w_weight_decay ${w_decay} \
########### Train mask setting 2 (currently used best) ###########
#### Added distil
nohup python -u train_autoprune_wgt_relu.py --gpu ${gpu} --arch resnet18_in --w_mask_lr ${w_mask_lr} --w_lr ${w_lr}\
 --mask_epochs ${epochs} --epochs 0 --lamda_wgt ${lamda_wgt} --lamda_relu ${lamda_relu} --batch_size 256 --precision half --dataset cifar10 \
 --train_mask --weight_prune --relu_prune\
 --distil --teacher_arch resnet18_in --teacher_path ./train_cifar/resnet18_in__cifar10/epoch_400/best.pth.tar\
 --pretrained_path ./train_cifar/resnet18_in__cifar10/epoch_400/best.pth.tar\
 --optim cosine > ./out/resnet18_in_mask_train_lamda_wgt_${lamda_wgt}_lamda_relu_${lamda_relu}_w_decay_${w_decay}_epoch_${epochs}.txt &
### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
### 
####  --distil --teacher_arch resnet18_in --teacher_path ./train_cifar/resnet18_in__cifar100/epoch_400/best.pth.tar\
#### --pretrained_path
####   --enable_lookahead \


#

# ########### Finetune mask version 2-2 (currently used best)  ###########
# ### Added distillation, and start from mask training with distillation
# nohup python -u train_autoprune_wgt_relu.py --gpu ${gpu} --arch resnet18_in --w_mask_lr ${w_mask_lr} --w_lr ${w_lr}\
#  --distil --teacher_arch resnet18_in --teacher_path ./train_cifar/resnet18_in__cifar10/epoch_400/best.pth.tar\
#  --mask_epochs 0 --epochs ${finetune_epoch} --lamda_wgt ${lamda_wgt} --lamda_relu ${lamda_relu} --batch_size 512 --precision half --dataset cifar10 \
#  --finetune_mask --weight_prune --relu_prune --w_lr_finetune ${w_lr_finetune} \
#  --pretrained_path ./train_cifar_spwgt_sprelu/resnet18_in_resnet18_in_cifar10/lambda_wgt_${lamda_wgt}_lamda_relu_${lamda_relu}_epoch_${epochs}_distil_belta500/checkpoint_mask_train.pth.tar\
#  --optim cosine > ./out/resnet18_in_mask_train_lamda_wgt_${lamda_wgt}_lamda_relu_${lamda_relu}_w_decay_${w_decay}_epoch_${finetune_epoch}_finetune.txt &
# ### Location of checkpoint file, log file, and tb for tensorboard file are in folder: 
# ### 
# ####  --distil --teacher_arch resnet18_in --teacher_path ./train_cifar/resnet18_in__cifar100/epoch_400/best.pth.tar\
# #### --pretrained_path
# ####   --enable_lookahead \