python /home/gezi/mine/hasky/deepiu/tools/classification-txt2json.py $1 
python /home/gezi/mine/hasky/deepiu/tools/scene_eval.py --submit ${1/%.txt/.json} --ref /home/gezi/data2/data/ai_challenger/scene/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json

