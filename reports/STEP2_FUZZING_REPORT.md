# Step 2: Generative Config & CLI Testing Results

Command tested: soup train --config <mutated.yaml> --dry-run --yes

| Input Class & Mutation | Observed Behavior | Return Code | Last Output Line | Assessment |
| :--- | :--- | :---: | :--- | :--- |
| baseline_valid | SILENT_ACCEPTANCE | 0 | Config valid. Ready to train! | PASS (Expected valid baseline) |
| typo:quantizaton | CLEAR_ERROR | 1 | 'quantization_aware'? Refused. | PASS (Properly rejected) |
| typo:val_splt | CLEAR_ERROR | 1 | unknown config key 'data.val_splt' - did you mean 'val_split'? Refused. | PASS (Properly rejected) |
| typo:model | CLEAR_ERROR | 1 | unknown config key 'model' - refused. | PASS (Properly rejected) |
| typo:lerning_rate | CLEAR_ERROR | 1 | unknown config key 'training.lerning_rate' - use training.lr. Refused. | PASS (Properly rejected) |
| typo:epocs | CLEAR_ERROR | 1 | Refused. | PASS (Properly rejected) |
| typo:unknown_top_key | CLEAR_ERROR | 1 | unknown config key 'unknown_top_key' - refused. | PASS (Properly rejected) |
| wrong_type:training.epochs='three' | CLEAR_ERROR | 1 | as an integer | PASS (Properly rejected) |
| wrong_type:training.lora.r='sixty_four' | CLEAR_ERROR | 1 | string as an integer | PASS (Properly rejected) |
| wrong_type:data.val_split=True | CLEAR_ERROR | 1 | data -> val_split: Input should be less than or equal to 0.5 | PASS (Properly rejected) |
| wrong_type:training.lr={'val': 0.0001} | CLEAR_ERROR | 1 | training -> lr: Input should be a valid number | PASS (Properly rejected) |
| wrong_type:training.epochs=1.5 | CLEAR_ERROR | 1 | fractional part | PASS (Properly rejected) |
| wrong_type:base=12345 | CLEAR_ERROR | 1 | base: Input should be a valid string | PASS (Properly rejected) |
| wrong_type:training.epochs=True | SILENT_ACCEPTANCE | 0 | Config valid. Ready to train! | **DEFECT (Silent Acceptance)** |
| wrong_type:training='not_a_dict' | CLEAR_ERROR | 1 | training: Input should be a valid dictionary or instance of TrainingConfig | PASS (Properly rejected) |
| wrong_type:training.lr=[0.001] | CLEAR_ERROR | 1 | training -> lr: Input should be a valid number | PASS (Properly rejected) |
| wrong_type:data.val_split='ten_percent' | CLEAR_ERROR | 1 | a number | PASS (Properly rejected) |
| out_of_range:training.epochs=0 | CLEAR_ERROR | 1 | training -> epochs: Input should be greater than or equal to 1 | PASS (Properly rejected) |
| out_of_range:training.lora.r=0 | SILENT_ACCEPTANCE | 0 | Config valid. Ready to train! | **DEFECT (Silent Acceptance)** |
| out_of_range:training.epochs=-5 | CLEAR_ERROR | 1 | training -> epochs: Input should be greater than or equal to 1 | PASS (Properly rejected) |
| out_of_range:data.max_length=-128 | CLEAR_ERROR | 1 | data -> max_length: Input should be greater than or equal to 64 | PASS (Properly rejected) |
| out_of_range:data.val_split=1.5 | CLEAR_ERROR | 1 | data -> val_split: Input should be less than or equal to 0.5 | PASS (Properly rejected) |
| out_of_range:data.val_split=-0.1 | CLEAR_ERROR | 1 | data -> val_split: Input should be greater than or equal to 0 | PASS (Properly rejected) |
| out_of_range:training.batch_size=-2 | CLEAR_ERROR | 1 | training -> batch_size: training.batch_size must be >= 1 or 'auto'; got -2 | PASS (Properly rejected) |
| out_of_range:training.batch_size=0 | CLEAR_ERROR | 1 | training -> batch_size: training.batch_size must be >= 1 or 'auto'; got 0 | PASS (Properly rejected) |
| out_of_range:training.lora.r=-8 | CLEAR_ERROR | 1 | training -> lora -> r: Input should be greater than or equal to 0 | PASS (Properly rejected) |
| out_of_range:training.lr=0.0 | CLEAR_ERROR | 1 | training -> lr: Input should be greater than 0 | PASS (Properly rejected) |
| out_of_range:training.lr=-0.0001 | CLEAR_ERROR | 1 | training -> lr: Input should be greater than 0 | PASS (Properly rejected) |
| out_of_range:training.lr=inf | SILENT_ACCEPTANCE | 0 | Config valid. Ready to train! | **DEFECT (Silent Acceptance)** |
| out_of_range:training.lr=nan | CLEAR_ERROR | 1 | training -> lr: Input should be greater than 0 | PASS (Properly rejected) |
| empty:empty_file | CLEAR_ERROR | 1 | Config file is empty | PASS (Properly rejected) |
| empty:null_training_block | CLEAR_ERROR | 1 | training: Input should be a valid dictionary or instance of TrainingConfig | PASS (Properly rejected) |
| empty:empty_train_path | CLEAR_ERROR | 1 | Run with --verbose for the full traceback. | **DEFECT (Unhandled Exception / Poor UX)** |
| empty:null_base | CLEAR_ERROR | 1 | base: Input should be a valid string | PASS (Properly rejected) |
| empty:empty_base | SILENT_ACCEPTANCE | 0 | Config valid. Ready to train! | **DEFECT (Silent Acceptance)** |
| empty:comments_only | CLEAR_ERROR | 1 | Config file is empty | PASS (Properly rejected) |
| empty:empty_dict | CLEAR_ERROR | 1 | data: Field required | PASS (Properly rejected) |
| conflict:wandb_and_tracker | CLEAR_ERROR | 1 | Run: pip install "soup-cli[wandb]" | PASS (Properly rejected) |
| conflict:hf_resume_without_push_as | CLEAR_ERROR | 1 | --hf-resume requires --push-as <repo> | PASS (Properly rejected) |
| conflict:echo_trap_without_config_flag | CLEAR_ERROR | 1 | --echo-trap-tokenizer-aware requires training.echo_trap_enabled=true | PASS (Properly rejected) |
| conflict:val_split_leaves_zero_train_rows | CLEAR_ERROR | 1 | data -> val_split: Input should be less than or equal to 0.5 | PASS (Properly rejected) |
| conflict:resume_nonexistent_dir | CLEAR_ERROR | 1 | not exist | PASS (Properly rejected) |