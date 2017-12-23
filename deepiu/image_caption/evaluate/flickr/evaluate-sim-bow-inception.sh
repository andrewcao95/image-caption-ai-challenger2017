cp ../../prepare/bow/flickr/conf.py conf.py
source ../../prepare/bow/flickr/config 

python ./evaluate-sim.py \
  --print_predict=0 \
  --image_url_prefix='D:\data\image-text-sim\flickr\imgs\' \
  --valid_resource_dir $valid_output_path \
  --vocab=$train_output_path/vocab.bin \
  --num_records_file=$train_output_path/num_records.txt \
  --show_eval 1 \
  --batch_size 100 \
  --fixed_eval_batch_size 10 \
  --num_fixed_evaluate_examples 3 \
  --metric_eval_batch_size 250 \
  --num_evaluate_examples 10 \
  --keep_interval 1 \
  --num_negs 1 \
  --use_neg 1 \
  --debug 0 \
  --algo bow \
  --interval 100 \
  --eval_interval 500 \
  --pre_calc_image_feature 0 \
  --margin 0.5 \
  --feed_dict 0 \
  --seg_method en \
  --feed_single 0 \
  --combiner sum \
  --exclude_zero_index 1 \
  --dynamic_batch_length 1 \
  --model_dir=$1 \

