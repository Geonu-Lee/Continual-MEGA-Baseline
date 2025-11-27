
# train continual classes
python train_continual.py --task_id 1 \
--base_folder results/scenario3 \
--meta_file meta_files/scenario3_30classes_tasks.json \
--task_class_num "30classes" --gpu 0 \
--base_meta_file meta_files/scenario3_base.json