mkdir -p valid.atten.feature.inceptionV4

#python ./imgs2features.py \
#    --image_dir ./caption_validation_images_20170910 \
#    --feature_name Mixed_7b \
#    --image_checkpoint_file /home/gezi/data/image_model_check_point/inception_v4.ckpt \
#    | python ./merge-pic-feature.py caption_validation_annotations_20170910.txt > caption_validation_annotations_20170910.atten.feature.inceptionV4.txt

#rand.py ./caption_validation_annotations_20170910.atten.feature.txt >  ./caption_validation_annotations_20170910.atten.feature.rand.txt 
mv ./caption_validation_annotations_20170910.atten.feature.inceptionV4.txt ./caption_validation_annotations_20170910.atten.feature.inceptionV4.rand.txt 

cd ./valid.atten.feature.inceptionV4
ln -s ../caption_validation_annotations_20170910.atten.feature.inceptionV4.rand.txt .
split.py caption_validation_annotations_20170910.atten.feature.inceptionV4.rand.txt 
cd ..

