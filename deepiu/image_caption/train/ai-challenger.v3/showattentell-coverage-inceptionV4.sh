conf_path=./prepare/default/app-conf/ai-challenger/seq-basic-atten-inceptionV4/

cp $conf_path/conf.py .
source $conf_path/config  

##NOTICE image encoder is Memory at first version but performance worse then Rnn so by default use Rnn encoder 
##seems batch norm and drop out for train image layer to image emb can improve a bit(not worse) so by default set to 1 
model_dir=/home/gezi/new/temp/image-caption/ai-challenger/model.v3/showattentell.coverage.inceptionV4
##TODO now ok but will load two image model graph init in two session, too much gpu mem usage, so just set samll metric_eval_examples, 500 -> 200 
## and eval rank will be slow here for generative model so can just disable eval rank during training and set metric eval examples to 500 
#assistant_model_dir=/home/gezi/new/temp/image-caption/ai-challenger/model/bow
assistant_model_dir=''
mkdir -p $model_dir 
mkdir -p $model_dir/epoch 
cp $0 $model_dir
cp $dir/vocab* $model_dir
cp $dir/vocab* $model_dir/epoch

python ./train.py \
  --train_input $train_output_path/'train-*' \
  --valid_input $valid_output_path/'test-*' \
  --valid_resource_dir $valid_output_path \
  --vocab $dir/vocab.txt \
  --image_dir $win_image_dir \
  --label_file $valid_output_path/'image_label.npy' \
  --img2text $valid_output_path/'img2text.npy' \
  --text2id $valid_output_path/'text2id.npy' \
  --image_name_bin $valid_output_path/'image_names.npy' \
  --image_feature_bin $valid_output_path/'image_features.npy' \
  --num_records_file  $train_output_path/num_records.txt \
  --model_dir=$model_dir \
  --assistant_model_dir="$assistant_model_dir" \
  --assistant_rerank_num 10 \
  --algo show_and_tell \
  --image_encoder Rnn \
  --showtell_encode_scope encode \
  --showtell_decode_scope decode \
  --coverage_attention_wrapper 1 \
  --eval_rank 0 \
  --eval_translation 1 \
  --image_attention_size 64 \
  --image_endpoint_feature_name Mixed_7b \
  --pre_calc_image_feature 1 \
  --image_model InceptionV4 \
  --image_checkpoint_file='/home/gezi/data/image_model_check_point/inception_v4.ckpt' \
  --image_features_batch_norm 1 \
  --image_features_drop_out 1 \
  --greedy_decode_with_logprobs 1 \
  --num_sampled 0 \
  --log_uniform_sample 1 \
  --fixed_eval_batch_size 0 \
  --num_fixed_evaluate_examples 0 \
  --num_evaluate_examples 3 \
  --show_eval 1 \
  --train_only 0 \
  --metric_eval 1 \
  --monitor_level 2 \
  --no_log 0 \
  --batch_size=256 \
  --num_gpus 0 \
  --eval_batch_size 1000 \
  --min_after_dequeue 512 \
  --learning_rate=0.1 \
  --learning_rate_decay_factor=0 \
  --num_epochs_per_decay=10 \
  --learning_rate_values=0.1,0.05,0.005,0.001 \
  --learning_rate_epoch_boundaries=20,25,30 \
  --num_epochs=40 \
  --eval_interval_steps 500 \
  --metric_eval_interval_steps 500 \
  --save_interval_steps 1000 \
  --save_interval_epochs 1 \
  --num_metric_eval_examples 200 \
  --metric_eval_batch_size 200 \
  --max_texts=10000 \
  --margin 0.5 \
  --feed_dict 0 \
  --num_records 0 \
  --min_records 0 \
  --seg_method $online_seg_method \
  --feed_single $feed_single \
  --seq_decode_method greedy \
  --length_normalization_factor 0.25 \
  --keep_prob 1. \
  --scheduled_sampling_probability 0. \
  --beam_size 10 \
  --emb_dim 512 \
  --rnn_hidden_size 512 \
  --dynamic_batch_length 1 \
  --log_device 0 \
  --work_mode full \

  #--model_dir /home/gezi/data/image-text-sim/model/model.ckpt-387000 \
  #2> ./stderr.txt 1> ./stdout.txt
