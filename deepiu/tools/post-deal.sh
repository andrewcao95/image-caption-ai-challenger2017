export CUDA_VISIBLE_DEVICES=-1
post-deal.py ensemble.evaluate-inference.txt ensemble.evaluate-inference-postdeal.txt 
aichallenger-evaluate.py ensemble.evaluate-inference-postdeal.txt 
