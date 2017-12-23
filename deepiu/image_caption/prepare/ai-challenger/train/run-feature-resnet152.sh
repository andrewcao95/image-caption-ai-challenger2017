mkdir -p train.feature.resnet152

c0 py ./imgs2features.py \
  --image_dir ./caption_train_images_20170902 \
  --image_checkpoint_file /home/gezi/data/image_model_check_point/resnet_v2_152.ckpt \
  | py ./merge-pic-feature.py caption_train_annotations_20170902.txt \
  > caption_train_annotations_20170902.feature.resnet152.txt

rand.py ./caption_train_annotations_20170902.feature.resnet152.txt \
  >  ./caption_train_annotations_20170902.feature.resnet152.rand.txt 

cd ./train.feature.resnet152
ln -s ../caption_train_annotations_20170902.feature.resnet152.rand.txt .
split.py caption_train_annotations_20170902.feature.resnet152.rand.txt 
cd ..

