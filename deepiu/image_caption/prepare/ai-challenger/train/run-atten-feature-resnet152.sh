mkdir -p train.atten.feature.resnet152

python ./imgs2features.py \
    --image_dir ./caption_train_images_20170902 \
    --feature_name postnorm \
    --image_checkpoint_file /home/gezi/data/image_model_check_point/resnet_v2_152.ckpt \
    | python ./merge-pic-feature.py caption_train_annotations_20170902.txt \
    > caption_train_annotations_20170902.atten.feature.resnet152.txt

#rand.py ./caption_train_annotations_20170902.atten.feature.txt >  ./caption_train_annotations_20170902.atten.feature.rand.txt 
mv ./caption_train_annotations_20170902.atten.feature.resnet152.txt \
  ./caption_train_annotations_20170902.atten.feature.resnet152.rand.txt

cd ./train.atten.feature.resnet152
ln -s ../caption_train_annotations_20170902.atten.feature.resnet152.rand.txt .
split.py caption_train_annotations_20170902.atten.feature.resnet152.rand.txt 
cd ..


