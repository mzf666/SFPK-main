python main.py \
  --exp iterative \
  --seed 0 \
  --cuda_idx 1 \
  --num_workers 8 \
  --data imagenet \
  --model resnet50v2 \
  --bsz 256 \
  --prn_epochs 10 \
  --ft_epochs 50 \
  --fixed_lr_epochs 0 \
  --lr 1e-2 \
  --n_grad_accum 8 \
  --warmup_epochs 5 \
  --cold_start_lr 1e-4 \
  --sparsity 0.05 \
  --prn_scope global \
  --pruner SFPK \
  --N 150 \
  --r 5.5 \
  --sfpk_n_mask 10 \
  --sfpk_repl_mode exp \
  --sfpk_repl_lam 0.2 \
  --sfpk_repl_weighted \
  --sfpk_vote_ratio 1 \
  --sfpk_vote_mode soft
