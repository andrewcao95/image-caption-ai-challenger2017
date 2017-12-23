cp ./app-conf/ai-challenger/seq-basic/* .
source ./config
vocab_dir=$dir

cp ./app-conf/ai-challenger/fetch-finetune/* . 
source ./config 
mkdir -p $dir
cp $vocab_dir/'vocab.txt' $dir
cp $vocab_dir/vocab* $dir
sh ./prepare-train.sh 

