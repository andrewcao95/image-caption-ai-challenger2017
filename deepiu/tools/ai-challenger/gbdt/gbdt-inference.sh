gen-ensemble-feature.py . inference 0.1
sh ./add-test-feature.sh
python ./gbdt-predict.py 
python ./gbdt-choose-best.py 
caption-txt2json.py  ensemble.inference.gbdt_result.best.txt
