cd src/gui_g2



export TASK_ALGO=grpo
export TASK_TYPE=grounding
export TASK_MODEL=qwen25
export TASK_DATASET=data19


export DEBUG_MODE="true"
export WANDB_MODE=offline

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 生成 RUN_NAME
RUN_NAME="gaussian_${TASK_DATASET}"


# example: export DATA_PATH=data_config/rec_internvl.yaml
export DATA_PATH=your_yaml_path


export LOG_DIR=/logs/${RUN_NAME}_${GPU_TYPE}_${TIMESTAMP}/

export SAVE_PATH=/${RUN_NAME}_${DATASET}_${GPU_TYPE}/

export PYTHONPATH=src


# Model path
export CKPT_PATH=your_model_path

pip list
pip install transformers==4.49.0
pip install deepspeed==0.15.4
# bash ../setup.sh
pip install filelock

# 创建文件夹
mkdir -p "${LOG_DIR}"

export LOG_PATH="${LOG_DIR}/log_${TIMESTAMP}_out.txt"
export WANDB_DIR="${LOG_DIR}"

# mkdir -p ${LOG_PATH}
# 1 16   2 16 4 8
export N_NODE=1
export N_GPU_PER_NODE=8


echo "N_NODE: $N_NODE"
echo "N_GPU_PER_NODE: $N_GPU_PER_NODE"
echo "LOG_DIR: $LOG_DIR"
echo "TASK_MEMO: $TASK_MEMO"
echo "DATA_PATH: $DATA_PATH"
echo "SAVE_PATH: $SAVE_PATH"

{
    echo "N_NODE: $N_NODE"
    echo "N_GPU_PER_NODE: $N_GPU_PER_NODE"
    echo "LOG_DIR: $LOG_DIR"
    echo "TASK_MEMO: $TASK_MEMO"
    echo "DATA_PATH: $DATA_PATH"
    echo "SAVE_PATH: $SAVE_PATH"

} > "$LOG_PATH"


WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29500}
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
# GPU_COUNT=1

DISTRIBUTED_ARGS="
    --nproc_per_node $GPU_COUNT \
    --nnodes ${WORLD_SIZE} \
    --node_rank ${RANK} \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS src/open_r1/gaussian_grpo.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir ${SAVE_PATH} \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --image_root <your_image_root> \
    --max_prompt_length 12048 \
    --num_generations 8 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to tensorboard \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 400 \
    --max_pixels 12845056 \
    --save_only_model true \
    --beta 0.04  \
    --learning_rate 1e-6 $@ 2>&1 | tee "${LOG_DIR}/log_${TIMESTAMP}.log"