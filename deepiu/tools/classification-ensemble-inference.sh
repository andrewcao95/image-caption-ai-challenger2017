#you should at first for each model do caption-inference.py to geneate model..inference.txt 
classification-ensemble-inference.py ./ inference  $1
classification-txt2json.py  ./ensemble.inference.txt 
