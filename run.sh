# This is an example for running the preprocessing, training and testing pipeline.

# 1. Please export ROIs annotations as GeoJson from QuPath.

# 2. Slide-Leel Stain Normalization Matrix Estimation
python calc_stain_normalization_reference.py --slides-dir "/data/train_val_slides/" --annotations-dir "/data/annotations/" --output-dir "assets/stain_norm/" --image-extension .ndpi --annotation-type tissue --annotation-extension .geojson --quality-control --method vahadane --slide-image-num 100 --dataset-image-num 0 --slide-downsample 2 --config "assets/preprocess_config.yml" --seed 42 --num-workers-loader $CPUS_PER_GPU --num-workers $SLURM_ARRAY_TASK_COUNT --worker-id $SLURM_ARRAY_TASK_ID

# 3. Dataset-Level Stain Normalization Matrix Estimation
python calc_stain_normalization_reference.py --slides-dir "/data/train_val_slides" --annotations-dir "/data/annotations/" --output-dir "/assets/stain_norm/train_val" --image-extension .ndpi --annotation-type tissue --annotation-extension .geojson --quality-control --method vahadane --slide-image-num 0 --dataset-image-num 3025 --dataset-downsample 4 --config "assets/preprocess_config.yml" --seed 42 --num-workers-loader 0 --num-workers $SLURM_ARRAY_TASK_COUNT --slide-reference-df "assets/stain_norm/vahadane/stain_vectors_slide_level_reference.csv"

# 4. Preprocess and extract tiles
python extract_preprocess_multires_tiles.py --slides-dir "/data/train_val_slides" --annotations-dir "/data/annotations/" --output-dir "/data/train_val_hdf5_384/" --slide-extension .ndpi --annotation-type tissue --ann-extension .geojson --labels-csv "ground_truth.csv" --image-col "slide_name" --label-col "tp53" --save-tiles --gpu ${SLURM_ARRAY_TASK_ID} --num-gpus ${SLURM_ARRAY_TASK_COUNT} --seed 42 --config "assets/preprocess_config.yml"

# 5. Extract additional features
python extract_additional_features.py --dataset-dir "/data/train_val_hdf5_384/" --gpu ${SLURM_ARRAY_TASK_ID} --num-gpus ${SLURM_ARRAY_TASK_COUNT} --seed 42 --config "assets/preprocess_config.yml"

# 6. Train model
python -u train.py -o "/output_dir/" --num-gpus-per-node $NUM_GPUS --num-nodes $NUM_NODES --config "assets/multires_study/tp53/clam/concat.yml" --name "EXPERIMENT_NAME" --fold $FOLD --run 1

# 7. Test model
python test.py --ckpt "/output_dir/EXPERIMENT_NAME"  --config "assets/multires_study/tp53/clam/concat.yml" --mode "test"