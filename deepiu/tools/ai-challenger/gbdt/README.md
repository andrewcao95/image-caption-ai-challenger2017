first run evaluate inference for 19 generative show atten tell models get 19 .evaluate-inference.txt 
run infernce for 19 generative models get 19 .inference.txt 
ensemble evaluate (ensemble-evaluate.sh) for the 19 models all weights 1.0 get ./ensemble.evaluate-inference.txt
ensemble inference (ensemble-inference.sh) for the 19 models all weights 1.0 get ./ensemble.inference.txt

cp ./ensemble.evaluate-inference.txt to ./ensemble.nomil.evaluate-inference.txt 
cp ./ensemble.inference.txt to ./ensemble.nomil.inference.txt 

ln mil model here
run mil evaluate inference for mil model with ./ensemble.nomil.evaluate-inference.txt to get ./ensemble.evaluate-inference.txt 
run mil inference for mil model with ./ensemble.nomil.inference.txt to get ./ensemble.inference.txt 
run ensemble-evaluate.sh again for 20 models (19 generative models + 1 mil model) to get ./ensemble.evaluate-inference.txt 
run ensemble-inference.sh again for 20 models to get ./ensemble.inference.txt 

run ai-best-evaluate.py ensemble.evaluate-inference.txt  to get upperbound for the 19 models generated captions  and also get all candidates captions segged info, cider and bleu4 score 

gen-ensemble-inference.sh for valid and test  to get basic gbdt model features


prepare to add addtional features

prepare-valid.sh 
do detection for valid and test data using google openimage faster-rcnn model 
do scene detection using classifier train on ai chanllenge scene classification contest remove about 1.2k images duplicate with caption valid images (iception resnet v2 model) for valid and test 
do lm score for all candidates caption both valid and test using model train on caption data using train captions only(no image)
dump attention 
dump logprobs
gen attention feature
gen logprobs feature
gen imagenet feature

prepare-test.sh

./run-cross-valid-rank.sh 
./gbdt-inference.sh 
