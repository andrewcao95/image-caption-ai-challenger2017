input=$(readlink -f $1)
echo $input

bsize=$2
lnf=$3

pushd .
cd /home/gezi/mine/hasky/deepiu/image_caption 
sh ./train/ai-challenger/showattentell-finetune-update-model.sh $input $bsize $lnf
popd
