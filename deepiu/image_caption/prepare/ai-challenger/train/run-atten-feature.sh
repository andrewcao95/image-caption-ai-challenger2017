mkdir -p train.atten.feature

python ./imgs2features.py \
    --image_dir ./caption_train_images_20170902 \
    --feature_name Conv2d_7b_1x1 \
    --image_checkpoint_file /home/gezi/data/image_model_check_point/inception_resnet_v2_2016_08_30.ckpt \
    | python ./merge-pic-feature.py caption_train_annotations_20170902.txt > caption_train_annotations_20170902.atten.feature.txt

#rand.py ./caption_train_annotations_20170902.atten.feature.txt >  ./caption_train_annotations_20170902.atten.feature.rand.txt 
mv ./caption_train_annotations_20170902.atten.feature.txt  ./caption_train_annotations_20170902.atten.feature.rand.txt

cd ./train.atten.feature
ln -s ../caption_train_annotations_20170902.atten.feature.rand.txt .
split.py caption_train_annotations_20170902.atten.feature.rand.txt 
cd ..


