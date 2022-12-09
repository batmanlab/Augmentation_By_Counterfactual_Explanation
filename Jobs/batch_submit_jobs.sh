# Classifier training
#!/bin/bash
partition=BatComputer
num_nodes=1
gpu=gpu:rtx6000:1
wall_time=18:00:00 # for 200k samples of size 128x128 1 hour for 2 epochs
mail_type=ALL

# user hyperparams
code_path=/jet/home/nmurali/asc170022p/nmurali/projects/augmentation_by_explanation_eccv22/Train_Classifier_DenseNet.py
config_path=/jet/home/nmurali/asc170022p/nmurali/projects/shortcut_detection_and_mitigation/experiments/medical_expts/chex/temp.yaml
job_name=chex_pneumonia

slurm_output=/jet/home/nmurali/asc170022p/nmurali/projects/augmentation_by_explanation_eccv22/Jobs/slurm_output/$job_name.stdout
slurm_err=/jet/home/nmurali/asc170022p/nmurali/projects/augmentation_by_explanation_eccv22/Jobs/slurm_output/$job_name.stderr
RUN_CMD="python $code_path --config $config_path"
sbatch -A bio170034p -p $partition -N $num_nodes --gres=$gpu -t $wall_time --mail-type $mail_type --mail-user nmurali --output=$slurm_output --error=$slurm_err --job-name=$job_name ./submit_job.sh $RUN_CMD




# # GAN training
# #!/bin/bash
# partition=BatComputer
# num_nodes=1
# gpu=gpu:rtx5000:1
# wall_time=48:00:00
# mail_type=ALL

# # user hyperparams
# code_path=/jet/home/nmurali/asc170022p/nmurali/projects/augmentation_by_explanation_eccv22/Explainer_StyleGANv2/Train_Explainer_StyleGANv2.py
# config_path=/jet/home/nmurali/asc170022p/nmurali/projects/augmentation_by_explanation_eccv22/Configs/Explainer/styleGAN_Skin_ln0p50.yaml
# job_name=gan_ham_ln0p50



# # Classifier (Energy-based OOD) training
# !/bin/bash
# partition=BatComputer
# num_nodes=1
# gpu=gpu:rtx5000:1
# wall_time=10:00:00
# mail_type=ALL

# # user hyperparams
# code_path=/jet/home/nmurali/asc170022p/nmurali/projects/augmentation_by_explanation_eccv22/Misc/energy_ood/CIFAR/train.py
# dataset=skin
# score=energy
# params=(0 1 2)

# for param in ${params[@]}
# do
#         slurm_output=/jet/home/nmurali/asc170022p/nmurali/projects/augmentation_by_explanation_eccv22/Jobs/slurm_output/$dataset$score$param.stdout
#         slurm_err=/jet/home/nmurali/asc170022p/nmurali/projects/augmentation_by_explanation_eccv22/Jobs/slurm_output/$dataset$score$param.stderr
#         RUN_CMD="python $code_path --dataset $dataset --seed $param --score $score"
#         sbatch -A bio170034p -p $partition -N $num_nodes --gres=$gpu -t $wall_time --mail-type $mail_type --mail-user nmurali --output=$slurm_output --error=$slurm_err --job-name=$dataset$score$param ./submit_job.sh $RUN_CMD
# done






