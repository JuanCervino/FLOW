#P_FL

# python run_FL.py  --homo_ratio 0.1 --n_workers_per_round 50 --reduce_to_ratio .2\
#                   --learner fed-pd --local_lr 1e-1 --local_epoch 5 --client_step_per_epoch 20\
#                   --eta 100 --use_ray --fed_pd_dual_lr 1 --imbalance
# No Imbalance
#   Reduce Classes not needed
#   Reduce to ratio not needed

# python run_FL_MCR.py  --homo_ratio 1 --n_workers_per_round 100 --dataset cifar10\
#                       --learner fed-avg --local_lr 1e-1  --client_step_per_epoch 1 --local_epoch 3\
#                       --heterogeneity dir --n_workers 100 --use_ray\
#                       --dir_level 5 --loss_fn mcr --fd 128 --model resnet\
#                       --n_global_rounds 502

# python run_FL_MCR.py  --homo_ratio 1 --n_workers_per_round 100 --dataset cifar10\
#                       --learner fed-avg --local_lr 1e-1  --client_step_per_epoch 1 --local_epoch 5\
#                       --heterogeneity dir --n_workers 100 --use_ray\
#                       --dir_level 5 --loss_fn mcr --fd 128 --model resnet\
#                       --n_global_rounds 502


# python run_FL_MCR.py  --homo_ratio 1 --n_workers_per_round 100 --dataset cifar10\
#                       --learner fed-avg --local_lr 3e-1  --client_step_per_epoch 1 --local_epoch 3\
#                       --heterogeneity dir --n_workers 100 --use_ray\
#                       --dir_level 5 --loss_fn mcr --fd 128 --model resnet\
#                       --n_global_rounds 502

# python run_FL_MCR.py  --homo_ratio 1 --n_workers_per_round 100 --dataset cifar10\
#                       --learner fed-avg --local_lr 3e-1  --client_step_per_epoch 1 --local_epoch 5\
#                       --heterogeneity dir --n_workers 100 --use_ray\
#                       --dir_level 5 --loss_fn mcr --fd 128 --model resnet\
#                       --n_global_rounds 502

python run_FL_MCR.py  --homo_ratio 1 --n_workers_per_round 100 --dataset cifar10\
                      --learner fed-avg --local_lr 4e-1  --client_step_per_epoch 1 --local_epoch 5\
                      --heterogeneity dir --n_workers 100 --use_ray\
                      --dir_level 5 --loss_fn mcr --fd 128 --model resnet\
                      --n_global_rounds 502 --remove_classes

python run_FL_MCR.py  --homo_ratio 1 --n_workers_per_round 100 --dataset cifar10\
                      --learner fed-avg --local_lr 5e-1  --client_step_per_epoch 1 --local_epoch 5\
                      --heterogeneity dir --n_workers 100 --use_ray\
                      --dir_level 5 --loss_fn mcr --fd 128 --model resnet\
                      --n_global_rounds 502 --remove_classes


python run_FL_MCR.py  --homo_ratio 1 --n_workers_per_round 100 --dataset cifar10\
                      --learner fed-avg --local_lr 3e-1  --client_step_per_epoch 1 --local_epoch 4\
                      --heterogeneity dir --n_workers 100 --use_ray\
                      --dir_level 5 --loss_fn mcr --fd 128 --model resnet\
                      --n_global_rounds 500 --remove_classes

python run_FL_MCR.py  --homo_ratio 1 --n_workers_per_round 100 --dataset cifar10\
                      --learner fed-avg --local_lr 4e-1  --client_step_per_epoch 1 --local_epoch 4\
                      --heterogeneity dir --n_workers 100 --use_ray\
                      --dir_level 5 --loss_fn mcr --fd 128 --model resnet\
                      --n_global_rounds 500 --remove_classes