python main.py \
  --exp oneshot \
  --seed 0 \
  --cuda_idx 0 \
  --num_workers 8 \
  --data cifar100 \
  --bsz 64 \
  --prn_epochs 1 \
  --ft_epochs 100 \
  --fixed_lr_epochs 0 \
  --lr 1e-2 \
  --warmup_epochs 5 \
  --cold_start_lr 1e-4 \
  --model resnet20 \
  --sparsity 0.05 \
  --prn_scope global \
  --pruner SFPK \
  --N 100 \
  --r 1.4 \
  --sfpk_n_mask 10 \
  --sfpk_repl_mode exp \
  --sfpk_repl_lam 0.2 \
  --sfpk_repl_weighted \
  --sfpk_vote_ratio 1 \
  --sfpk_vote_mode soft
