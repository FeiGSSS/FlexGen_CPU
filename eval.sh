$MODEL="facebook/opy-125m" # model_name_in_huggingface
$MODEL_TYPE="flexgen"
$PATH="" # your_model_path
$TASKS="toxigen" #  which_task_you_want_to_eval, default_to_toxigen
$OUTPUT_PATH="eval_out/test" # which_path_you_want_to_save_results
$BATCH_SIZE=1



python eval_utils \
    --model $MODEL \
    --model_type $MODEL_TYPE\
    --path $PATH \
    --tasks $TASKS  \
    --output_path $OUTPUT_PATH \
    --batch_size $BATCH_SIZE \
    --log_sample 