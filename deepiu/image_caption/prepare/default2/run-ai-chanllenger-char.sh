cp ./app-conf/ai-challenger/char/* .
sh ./run-local.sh 

source ./config
vocab_dir=$dir

cp ./app-conf/ai-challenger/char-atten/* .
source ./config 
mkdir -p $dir
cp $vocab_dir/'vocab.txt' $dir
sh ./run-local-novocab.sh 
cp $vocab_dir/valid/all_refs.pkl $dir/valid 
cp $vocab_dir/valid/valid_refs.pkl $dir/valid 

cp ./app-conf/ai-challenger/char-finetune/* . 
source ./config 
mkdir -p $dir
cp $vocab_dir/'vocab.txt' $dir
sh ./run-local-novocab.sh 
cp $vocab_dir/valid/all_refs.pkl $dir/valid 
cp $vocab_dir/valid/valid_refs.pkl $dir/valid 

