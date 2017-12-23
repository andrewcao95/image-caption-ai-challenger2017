$result_file=$1.evaluate-inference.txt
if [ ! -f "$result_file"  ]; then
  python ./inference/ai-challenger/evaluate-inference.py $1 
fi

$result_file=$1.caption_metrics.txt
if [ ! -f "$result_file"  ]; then
  python ./evaluate/ai-challenger/evaluate.py $1.evaluate-inference.txt
fi
