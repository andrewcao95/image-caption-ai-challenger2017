mkdir -p valid.atten.feature.resnet152

python ./imgs2features.py \
    --image_dir ./caption_validation_images_20170910 \
    --feature_name postnorm \
    --image_checkpoint_file /home/gezi/data/image_model_check_point/resnet_v2_152.ckpt \
    | python ./merge-pic-feature.py caption_validation_annotations_20170910.txt > caption_validation_annotations_20170910.atten.feature.resnet152.txt

#rand.py ./caption_validation_annotations_20170910.atten.feature.txt >  ./caption_validation_annotations_20170910.atten.feature.rand.txt 
mv ./caption_validation_annotations_20170910.atten.feature.resnet152.txt ./caption_validation_annotations_20170910.atten.feature.resnet152.rand.txt 

cd ./valid.atten.feature.resnet152
ln -s ../caption_validation_annotations_20170910.atten.feature.resnet152.rand.txt .
split.py caption_validation_annotations_20170910.atten.feature.resnet152.rand.txt 
cd ..
