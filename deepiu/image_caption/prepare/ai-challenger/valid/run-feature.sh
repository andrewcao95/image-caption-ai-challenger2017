mkdir -p valid.feature

py ./imgs2features.py \
  --image_dir ./caption_validation_images_20170910 \
  --image_checkpoint_file /home/gezi/data/image_model_check_point/inception_resnet_v2_2016_08_30.ckpt \
  | py ./merge-pic-feature.py \
  caption_validation_annotations_20170910.txt > caption_validation_annotations_20170910.feature.txt

rand.py ./caption_validation_annotations_20170910.feature.txt >  ./caption_validation_annotations_20170910.feature.rand.txt 

cd ./valid.feature
ln -s ../caption_validation_annotations_20170910.feature.rand.txt .
split.py caption_validation_annotations_20170910.feature.rand.txt 
cd ..

