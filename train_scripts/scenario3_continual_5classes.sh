
# train continual classes
for i in $(seq 1 6); do
    python train_continual.py --task_id $i \
    --base_folder results/scenario3 \
    --meta_file meta_files/scenario3_5classes_tasks.json \
    --task_class_num "5classes" --gpu 0 \
    --base_meta_file meta_files/scenario3_base.json
done