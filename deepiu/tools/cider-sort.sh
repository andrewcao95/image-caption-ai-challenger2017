echo 'gbdt:'
grep $1 ensemble.gbdt.caption_metrics.txt | head -3
echo 'pre gbdt:'
grep $1 ensemble.best_metrics.txt | head -3
echo 'dream:'
grep $1 ensemble.best_metrics.txt | sort.py --num_lines=5 3
echo 'info:'
grep $1 ensemble.caption_metrics.txt 
