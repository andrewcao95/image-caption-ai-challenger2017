source ./prepare/default/app-conf/ai-challenger/seq-basic-finetune/config 
cp ./prepare/default/app-conf/ai-challenger/seq-basic-finetune/conf.py .

dir=/home/gezi/new/temp/image-caption/ai-challenger/v1
model_dir=$dir/model/exp
dir=$dir/tfrecord/seq-basic-finetune

evaluate-model.py $model_dir \
  --vocab $dir/vocab.txt \
  --eval_rank=0 \
  --num_metric_eval_examples $1 \
  --metric_eval_batch_size 500 \
  --valid_resource_dir $dir/valid \
  --caption_file $model_dir/caption_eval.txt \
  --caption_metrics_file $model_dir/caption_metrics.txt 
