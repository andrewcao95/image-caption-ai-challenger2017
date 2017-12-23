# inception resnet v2
ensemble-ln.py ../../showattentell.coverage.finetune.lr0005/epoch/model.ckpt-20.90-685729 
ensemble-ln.py ../../showattentell.coverage.finetune.lr0005/epoch/model.ckpt-24.40-800564

ensemble-ln.py ../../showattentell.bahdanau.finetune/epoch/model.ckpt-26.70-876027
ensemble-ln.py ../../showattentell.bahdanau.finetune/epoch/model.ckpt-28.70-941647 

ensemble-ln.py ../../showattentell.luong.finetune/epoch/model.ckpt-35.90-1177879
ensemble-ln.py ../../showattentell.luong.finetune/epoch/model.ckpt-42.50-1394425 

# inception v4 
ensemble-ln.py ../../showattentell.coverage.inceptionV4.finetune/epoch/model.ckpt-34.10-1118821
ensemble-ln.py ../../showattentell.coverage.inceptionV4.finetune/epoch/model.ckpt-38.10-1250061 

ensemble-ln.py ../../showattentell.bahdanau.inceptionV4.finetune/epoch/model.ckpt-28.00-918680
ensemble-ln.py ../../showattentell.bahdanau.inceptionV4.finetune/epoch/model.ckpt-37.10-1217251 

ensemble-ln.py ../../showattentell.luong.inceptionV4.finetune/epoch/model.ckpt-27.80-912118 
ensemble-ln.py ../../showattentell.luong.inceptionV4.finetune/epoch/model.ckpt-36.80-1207408 

# nasnet 
ensemble-ln.py ../../showattentell.coverage.nasnet.finetune.later.lr01/epoch/model.ckpt-16.40-538084
ensemble-ln.py ../../showattentell.coverage.nasnet.finetune.later.lr01/epoch/model.ckpt-18.40-603704 

ensemble-ln.py ../../showattentell.bahdanau.nasnet.finetune.2gpu.lr0005/epoch/model.ckpt-13.30-436373
ensemble-ln.py ../../showattentell.bahdanau.nasnet.finetune.2gpu.lr0005/epoch/model.ckpt-15.30-501993
ensemble-ln.py ../../showattentell.bahdanau.nasnet.finetune.2gpu.lr0005/epoch/model.ckpt-17.30-567613 

ensemble-ln.py ../../showattentell.luong.nasnet.finetune.later.lr01/epoch/model.ckpt-15.60-511836
ensemble-ln.py ../../showattentell.luong.nasnet.finetune.later.lr01/epoch/model.ckpt-19.80-649638 

# got mil.idf.rnn2  
# c3 evaluate-mil-inference-evaluate.py model.ckpt-27.00-22140 /home/gezi/mine/mount/temp/image-caption/ai-challenger/model.v4/ensemble/20171126/ensemble.evaluate-inference.txt;c3 run-mil-inference.py model.ckpt-27.00-22140 /home/gezi/mine/mount/temp/image-caption/ai-challenger/model.v4/ensemble/20171126/ensemble.inference.txt 

# mil 
ensemble-ln.py ../../mil.idf.rnn2/epoch/model.ckpt-27.00-22140

