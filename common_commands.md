# 移动生成数据备份

mv data/rpy_grid_motion_safety/rpy_grid_motion_safety_clean \
   data/rpy_grid_motion_safety/rpy_grid_motion_safety_clean.bak_$(date +%Y%m%d_%H%M%S)

# ["inspect", "convert", "infer", "compare", "all"]

CUDA_VISIBLE_DEVICES=1 /data1/liuwenhao/conda/envs/BWM/bin/python robotwin_to_bwm.py \
   --stage all \
   --robotwin_dir third_party/robotwin/data/beat_block_hammer/demo_clean \
   --output_dir outputs/robotwin_bwm/beat_block_hammer_demo_clean \
   --bwm_root third_party/boundless-world-model \
   --num_episodes 50 \
   --overwrite


# 仿真执行
cd /data1/liuwenhao/Projects/robotwin_consistency/third_party/robotwin

TORCH_CUDA_ARCH_LIST=12.0 \
TORCH_EXTENSIONS_DIR=/data1/liuwenhao/tmp/torch_extensions \
bash collect_data.sh rpy_grid_motion_safety rpy_grid_motion_safety_clean 1 --denoiser oidn


cp task_config/rpy_grid_motion_safety_clean.yml task_config/rpy_grid_motion_safety_clean_13_left.yml

# 测试可达性
python rpy_direction_validation/run_grid_motion_reachability.py \
   --gpu 1 \
   --workers 4 \
   --seed 0 \
   --denoiser oidn

# 真实执行
python rpy_direction_validation/run_grid_rpy_rotation_shards.py \
   --gpu 1 \
   --workers 4 \
   --seed 0 \
   --denoiser oidn

