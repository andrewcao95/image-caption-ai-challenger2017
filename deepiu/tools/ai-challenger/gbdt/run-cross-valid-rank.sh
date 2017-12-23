gen-ensemble-feature.py . eval 0.1
merge-rank-label.py 
sh ./add-feature.sh
#ln -s ensemble.train.detect.scene.lm.txt ensemble.train.final.txt
sh ./cross-valid-rank.sh 
calc-rank-score.py 
#nc aichallenger-evaluate.py ensemble.gbdt.evaluate-inference.txt 
