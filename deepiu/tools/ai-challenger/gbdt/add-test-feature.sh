python ./gen-detection-feature.py --type=test
python ./gen-scene-feature.py --type=test
python ./gen-lm-feature.py  --type=test 
ln -s ./ensemble.inference.feature.detect.scene.lm.txt ./ensemble.inference.feature.final.txt 
add-fake-rank-label.py < ./ensemble.inference.feature.final.txt > ./ensemble.inference.feature.final.fakelabel.txt 
mlt ensemble.inference.feature.final.fakelabel.txt --name 0,1 -group 0  -c si
mlt ensemble.inference.feature.final.fakelabel.txt --name 0,1 -group 0  -c fss -vl 1

