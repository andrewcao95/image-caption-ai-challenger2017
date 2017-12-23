mlt ensemble.train.final.txt --name 0,1 -group 0  -c si
mlt ensemble.train.final.txt --name 0,1 -group 0  -c fss -vl 1
#mlt ensemble.train.txt --name 0,1 -group 0  -cl gbrank -wr 1 -excl model,vote,tfidf --ntree 10 --mil 100 -smooth 0.5 --entropy 0.5 -nbag 10 -nbagfrac 0.8 -nt 10
#mlt ensemble.train.detect_person.txt --name 0,1 -group 0  -cl gbrank -wr 1 -excl model,vote,tfidf -smooth 0.5 --entropy 0.5 -nbag 10 -nbagfrac 0.8 -nt 10 --ntree 20 --nl 32 --lr 0.1 --mil 100
#mlt ensemble.train.detect_person.txt --name 0,1 -group 0  -cl gbrank -wr 1 -excl model,vote,tfidf -smooth 0.5 --entropy 0.5 -nbag 10 -nbagfrac 0.8 -nt 10 -mil 100  -ntree 100 --nl 128 --lr 0.025 
#--below now cider 1.9208
#mlt ensemble.train.detect.txt --name 0,1 -group 0  -cl gbrank -wr 1 -excl model,vote,tfidf -smooth 0.5 --entropy 0.5 -nt 10 -mil 100  -ntree 100 --nl 128 --lr 0.025 
#mlt ensemble.train.detect.txt --name 0,1 -group 0  -cl gbrank -wr 1 -excl model,vote,tfidf -smooth 0.5 --entropy 0.5 -nt 10 -mil 100  -ntree 100 --nl 128 --lr 0.025 
#mlt ensemble.train.final.txt --rank=1  -cl gbrank -wr 1 -smooth 0.5 --entropy 0.5 -nt 12 -mil 100  -ntree 500 --nl 128 --lr 0.025  -maxfs 40 -rs 1138636380 excl model,vote,tfidf,phone,person,entropy,coverage

#mlt ensemble.train.final.txt --rank=1  -cl gbrank -wr 1 -nt 12 -mil 100  -ntree 500 --nl 128 --lr 0.025  -maxfs 60 -rs 1138636380 -excl model,vote,tfidf,phone,attention_coverage_loss,entropy,detect_num,detect_person,face_pos,logprobs_sum -ntree 500
#2017-12-09 18:19:51 0:01:31 trans_avg:[0.9213]	trans_bleu_1:[0.8556]	trans_bleu_2:[0.7661]	trans_bleu_3:[0.6818]	trans_bleu_4:[0.6050]	trans_cider:[1.9598]	trans_meteor:[0.4200]	trans_rouge_l:[0.7004]	ensemble.gbdt.evaluate-inference.txt

mlt ensemble.train.final.txt --rank=1  -cl gbrank -wr 1 -smooth 0.5 --entropy 0.5 -nt 12 -mil 100  -ntree 500 --nl 128 --lr 0.025  -maxfs 60 -rs 1138636380 -excl model,vote,tfidf,phone,attention_coverage_loss,entropy,detect_num,detect_person,face_pos,logprobs_sum -ntree 500
#2017-12-05 22:06:07 0:01:31 trans_avg:[0.9213]	trans_bleu_1:[0.8558]	trans_bleu_2:[0.7663]	trans_bleu_3:[0.6821]	trans_bleu_4:[0.6054]	trans_cider:[1.9592]	trans_meteor:[0.4201]	trans_rouge_l:[0.7005]	ensemble.gbdt.evaluate-inference.txt

calc-rank-score.py 
nc aichallenger-evaluate.py ensemble.gbdt.evaluate-inference.txt 
