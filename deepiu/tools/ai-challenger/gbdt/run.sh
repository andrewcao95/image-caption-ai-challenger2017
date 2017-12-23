gen-ensemble-feature.py . eval 
merge-rank-label.py
sh ./train.sh 
calc-rank-score.py 
nc aichallenger-evaluate.py ensemble.gbdt.evaluate-inference.txt 
