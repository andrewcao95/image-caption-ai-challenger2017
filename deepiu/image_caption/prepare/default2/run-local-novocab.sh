source ./config 
ln -s $resource_dir/data .
ln -s $resource_dir/conf .

echo 'From train data dir:', $train_data_path
echo 'Will write to train output dir:', $train_output_path

#echo '-------gen vocab'
#sh ./gen-vocab.sh 

echo '-------prepare fixed valid'
sh ./prepare-fixed-valid.sh

echo '-------prepare valid'
sh ./prepare-valid.sh

echo '-------prepare train'
sh ./prepare-train.sh  


#rm data 
#rm conf 
