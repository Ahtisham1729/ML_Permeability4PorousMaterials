#!/bin/bash
#SBATCH --job-name=lbm_simulation
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --array=0-5720%64
#SBATCH --output=logs_256/lbm_task_%a.out
#SBATCH --error=logs_256/lbm_task_%a.err

# Configuration
INPUT_DIR="/data/cat/ws/ahkh532h-permeability/data_256/merged_single_connectivity_phase0"
OUTPUT_DIR="/data/cat/ws/ahkh532h-permeability/LBM_results/256"
SCRIPT_PATH="/data/cat/ws/ahkh532h-permeability/lbm-simulation/lbm_batch.py"
LOG_DIR="logs_256"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Array Job: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

module --force purge
module load release/24.04
module load GCC/12.3.0
module load Python/3.11.3
module load CUDA/12.6.0

cd /data/cat/ws/ahkh532h-permeability/lbm-simulation
source .venv/bin/activate

python -c "import cupy; a = cupy.zeros(10); cupy.cuda.Device().synchronize()"

export CUDA_LAUNCH_BLOCKING=1
export CUPY_CACHE_DIR=$TMPDIR/cupy_cache

echo "Processing sample index: $SLURM_ARRAY_TASK_ID"

# Run the simulation - output goes to SLURM log files
python "$SCRIPT_PATH" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --sample-index "$SLURM_ARRAY_TASK_ID"

EXIT_CODE=$?

echo ""
echo "Task $SLURM_ARRAY_TASK_ID finished: $(date)"
echo "Exit code: $EXIT_CODE"

exit $EXIT_CODE
