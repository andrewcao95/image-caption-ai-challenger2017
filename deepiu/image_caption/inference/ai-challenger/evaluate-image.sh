source ./prepare/default/app-conf/ai-challenger/seq-basic-finetune/config 
cp ./prepare/default/app-conf/ai-challenger/seq-basic-finetune/conf.py .

model_path=$1
model_dir=$(dirname "$model_path") 
caption-evaluate.py $model_path \
  --vocab $dir/vocab.txt \
  --eval_rank=0 \
  --num_metric_eval_examples $2 \
  --metric_eval_batch_size 500 \
  --valid_resource_dir $dir/valid \
  --caption_file $model_dir/caption_eval.txt \
  --caption_metrics_file $model_dir/caption_metrics.txt
