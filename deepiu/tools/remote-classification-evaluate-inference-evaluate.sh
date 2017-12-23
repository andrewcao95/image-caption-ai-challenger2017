base=/home/gezi/mine/hasky/deepiu/tools/
dir_=`pwd`
odir=${dir_/hdfsmount/mount}
odir=${odir/gpu1/gezi}
mkdir -p odir
ofile=$odir/$1
result=$ofile.evaluate-inference.txt 
touch $result
python $base/classification-evaluate-inference.py $1 $result

python $base/classification-txt2json.py $result
python $base/scene_eval.py --submit ${result/%.txt/.json} --ref /home/gezi/data2/data/ai_challenger/scene/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json

CUDA_VISIBLE_DEVICES=-1 python $base/classification-evaluate.py $result 
