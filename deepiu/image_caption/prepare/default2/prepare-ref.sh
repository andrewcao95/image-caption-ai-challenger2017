source ./config 

cat $valid_data_path/* | python ./gen-refs.py $valid_output_path/'valid_refs.pkl' $valid_output_path/'valid_ref_len.txt' $valid_output_path/'valid_refs_document_frequency.dill'

cat $valid_data_path/* $train_data_path/* | python ./gen-refs.py $valid_output_path/'all_refs.pkl' $valid_output_path/'all_ref_len.txt' $valid_output_path/'all_refs_document_frequency.dill'

