@echo off
setlocal

set MT_NAME=%1
set N_EXPERTS=%2

python ..\..\..\run_minigrid_ppo_mt_svd.py ^
  --use_cuda ^
  --env_name %MT_NAME% ^
  --exp_name ppo_mt_moore_singlehead_svd_gpu_%N_EXPERTS%e ^
  --n_exp 5 ^
  --n_epochs 100 ^
  --n_steps 2000 ^
  --n_episodes_test 16 ^
  --lr_actor 1e-3 ^
  --lr_critic 1e-3 ^
  --n_experts %N_EXPERTS% ^
  --train_frequency 2000 ^
  --orthogonal ^
  --actor_network MiniGridPPOMixtureSHNetwork_SVD ^
  --critic_network MiniGridPPOMixtureSHNetwork_SVD ^
  --actor_n_features 128 ^
  --critic_n_features 128 ^
  --batch_size 256 ^
  --gamma 0.99

endlocal