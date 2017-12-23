dir=/home/gezi/new/temp/image-caption/ai-challenger/
model_dir=$dir/model/showattentell.finetune

python /home/gezi/mine/hasky/deepiu/tools/caption-inference.py \
  --vocab=$dir/tfrecord/seq-basic-finetune/'vocab.txt' \
  --model_dir=$model_dir/epoch/model.ckpt-13.5-132000 \
  --test_image_dir=/home/gezi/new2/data/ai_challenger/ai_challenger_caption_test1_20170923/caption_test1_images_20170923/ \
  --buffer_size  500 > $model_dir/caption.txt

#after this you can use deepiu.tools/ caption-txt2json.py to convert to json file 
