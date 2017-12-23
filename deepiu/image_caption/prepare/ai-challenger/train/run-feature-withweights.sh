mkdir -p train.feature

cat ./caption_train_annotations_20170902.txt | python ./add-tfidf.py > caption_train_annotations_20170902.withweights.txt

#python ./imgs2features.py \
#    --image_dir ./caption_train_images_20170902 \
#    | python ./merge-pic-feature-withweights.py \
#    caption_train_annotations_20170902.withweights.txt \
#    > caption_train_annotations_20170902.feature.txt

cat img_feature.txt | python ./merge-pic-feature-withweights.py \
    caption_train_annotations_20170902.withweights.txt > caption_train_annotations_20170902.feature.txt

rand.py ./caption_train_annotations_20170902.feature.txt >  ./caption_train_annotations_20170902.feature.rand.txt 

cd ./train.feature
ln -s ../caption_train_annotations_20170902.feature.rand.txt .
split.py caption_train_annotations_20170902.feature.rand.txt 
cd ..

