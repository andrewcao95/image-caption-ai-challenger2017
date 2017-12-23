dir=/home/gezi/new/temp/image-caption/ai-challenger/
model_dir=$1

python /home/gezi/mine/hasky/deepiu/tools/caption-inference.py \
  --vocab=$dir/tfrecord/seq-basic-finetune/'vocab.txt' \
  --model_dir=$model_dir \
  --test_image_dir=/home/gezi/data2/data/ai_challenger/ai_challenger_caption_validation_20170910/caption_validation_images_20170910 \
  --buffer_size  500 > $model_dir/evaluate.inference.txt

#after this you can use deepiu.tools/ caption-txt2json.py to convert to json file 
