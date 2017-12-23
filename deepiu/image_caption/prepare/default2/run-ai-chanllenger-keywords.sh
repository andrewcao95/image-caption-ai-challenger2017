cp ./app-conf/ai-challenger/seq-basic/* .

source ./config
vocab_dir=$dir

cp ./app-conf/ai-challenger/seq-basic-atten-keywords/* .
source ./config 
mkdir -p $dir
cp $vocab_dir/'vocab.txt' $dir
cp $vocab_dir/vocab* $dir
sh ./run-local-novocab.sh 
cp $vocab_dir/valid/all_refs.pkl $dir/valid 
cp $vocab_dir/valid/valid_refs.pkl $dir/valid 
cp $vocab_dir/valid/valid_ref_len.txt $dir/valid 
cp $vocab_dir/valid/valid_refs_document_frequency.dill $dir/valid 
