mkdir -p valid.atten.feature

python ./imgs2features.py \
    --image_dir ./caption_validation_images_20170910 \
    --feature_name Conv2d_7b_1x1 \
    --image_checkpoint_file /home/gezi/data/image_model_check_point/inception_resnet_v2_2016_08_30.ckpt \
    | python ./merge-pic-feature.py caption_validation_annotations_20170910.txt > caption_validation_annotations_20170910.atten.feature.txt

#rand.py ./caption_validation_annotations_20170910.atten.feature.txt >  ./caption_validation_annotations_20170910.atten.feature.rand.txt 
mv ./caption_validation_annotations_20170910.atten.feature.txt ./caption_validation_annotations_20170910.atten.feature.rand.txt 

cd ./valid.atten.feature
ln -s ../caption_validation_annotations_20170910.atten.feature.rand.txt .
split.py caption_validation_annotations_20170910.atten.feature.rand.txt 
cd ..

