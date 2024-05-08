#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=explain_MNIST
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

source "/etc/slurm/local_job_dir.sh"


mkdir -p ${LOCAL_JOB_DIR}/outputs

tar -czf ${LOCAL_JOB_DIR}/config_files.tgz ${HOME}/DualView/config_files
tar -czf ${LOCAL_JOB_DIR}/checkpoints.tgz ${HOME}/DualView/checkpoints
echo "TAR DONE"
tar -C ${LOCAL_JOB_DIR} -zxf ${LOCAL_JOB_DIR}/config_files.tgz home/fe/yolcu/DualView/config_files --strip-components=4
tar -C ${LOCAL_JOB_DIR} -zxf ${LOCAL_JOB_DIR}/checkpoints.tgz home/fe/yolcu/DualView/checkpoints --strip-components=4


fname_config=$(basename "$1")
config_name=${fname_config::-5}

SECONDS=0

singularity \
  run \
        --nv \
        --bind ${LOCAL_JOB_DIR}/config_files:/mnt/config_files \
        --bind ${LOCAL_JOB_DIR}/checkpoints:/mnt/checkpoints \
        --bind ${DATAPOOL3}/datasets:/mnt/dataset \
        --bind ${LOCAL_JOB_DIR}/outputs:/mnt/outputs \
        ../singularity/explain.sif --config_file /mnt/config_files/cluster/explain/MNIST/${fname_config}

duration=$SECONDS
echo "The job took $(($duration / 60)) minutes and $(($duration%60)) seconds"

cd ${LOCAL_JOB_DIR}
tar -czf MNIST-${fname_config}-output_data.tgz outputs
cp MNIST-${fname_config}-output_data.tgz ${SLURM_SUBMIT_DIR}

rm -rf ${LOCAL_JOB_DIR}/*
