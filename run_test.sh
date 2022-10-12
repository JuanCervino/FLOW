




python run_FL_MCR.py  --homo_ratio 1 --n_workers_per_round 10 --dataset cifar100\
                      --learner fed-avg --local_lr 1e-1 --local_epoch 1 --client_step_per_epoch 5\
                      --use_gradient_clip --heterogeneity dir --n_workers 10\
                      --dir_level 5 --loss_fn mcr --fd 512 --model resnet\
                      --n_global_rounds 4