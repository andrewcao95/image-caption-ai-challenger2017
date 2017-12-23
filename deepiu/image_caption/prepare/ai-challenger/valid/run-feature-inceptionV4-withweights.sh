mkdir -p valid.feature.inceptionV4

python ./imgs2features.py \
    --image_dir ./caption_validation_images_20170910 \
    --image_checkpoint_file /home/gezi/data/image_model_check_point/inception_v4.ckpt \
    --batch_size_ 300 \
    | python ./merge-pic-feature-withweights.py caption_validation_annotations_20170910.withweights.txt \
    > caption_validation_annotations_20170910.feature.inceptionV4.txt

rand.py ./caption_validation_annotations_20170910.feature.inceptionV4.txt \
    >  ./caption_validation_annotations_20170910.feature.inceptionV4.rand.txt 

cd ./valid.feature.inceptionV4
ln -s ../caption_validation_annotations_20170910.feature.inceptionV4.rand.txt .
split.py caption_validation_annotations_20170910.feature.inceptionV4.rand.txt 
cd ..

