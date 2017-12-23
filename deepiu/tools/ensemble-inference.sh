#you should at first for each model do caption-inference.py to geneate model..inference.txt 
ensemble-inference.py ./ inference  $1
caption-txt2json.py ./ensemble.inference.txt ./ensemble.inference.json
#post-deal.py ./ensemble.inference.txt ./ensemble.inference.postdeal.txt 
#caption-txt2json.py ./ensemble.inference.postdeal.txt ./ensemble.inference.postdeal.json 
#inference-caption2html.py ./ensemble.inference.postdeal.txt
