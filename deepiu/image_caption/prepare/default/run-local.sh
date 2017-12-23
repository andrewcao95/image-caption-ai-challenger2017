source ./config 
ln -s $resource_dir/data .
ln -s $resource_dir/conf .

echo '-------gen vocab'
sh ./gen-vocab.sh 

sh ./run-local-novocab.sh 

echo '-------prepare ref'
sh ./prepare-ref.sh 

#rm data 
#rm conf 
