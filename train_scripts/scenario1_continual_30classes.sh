
# train continual classes
for i in $(seq 1 2); do
    python train_continual.py --task_id $i \
    --base_folder results/scenario1 \
    --meta_file meta_files/scenario1_30classes_tasks.json \
    --task_class_num "30classes" --gpu 0 \
    --base_meta_file meta_files/scenario1_base.json
done