mkdir -p train.feature.inceptionV4

py ./imgs2features.py \
  --image_dir ./caption_train_images_20170902 \
  --image_checkpoint_file /home/gezi/data/image_model_check_point/inception_v4.ckpt \
  | py ./merge-pic-feature-withweights.py caption_train_annotations_20170902.withweights.txt \
  > caption_train_annotations_20170902.feature.inceptionV4.txt

rand.py ./caption_train_annotations_20170902.feature.inceptionV4.txt \
  >  ./caption_train_annotations_20170902.feature.inceptionV4.rand.txt 

cd ./train.feature.inceptionV4
ln -s ../caption_train_annotations_20170902.feature.inceptionV4.rand.txt .
split.py caption_train_annotations_20170902.feature.inceptionV4.rand.txt 
cd ..

