source ./prepare/default/app-conf/ai-challenger/seq-basic-atten/config 
cp ./prepare/default/app-conf/ai-challenger/seq-basic-atten/conf.py .

model_path=$1
model_dir=$(dirname "$model_path") 

echo $dir

caption-evaluate.py $model_path \
  --vocab $dir/vocab.txt \
  --eval_rank=0 \
  --num_metric_eval_examples $2 \
  --metric_eval_batch_size 500 \
  --valid_resource_dir $dir/valid \
  --image_model InceptionResnetV2 \
  --image_checkpoint_file='/home/gezi/data/image_model_check_point/inception_resnet_v2_2016_08_30.ckpt' \
  --caption_file $model_dir/caption_eval.txt \
  --caption_metrics_file $model_dir/caption_metrics.txt
