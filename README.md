# image-caption-ai-challenger2017   
got 5# place for objective score and 2# place for subjective score  
NOTICE single model perform less good as using detection features based attention models(more detatil in more info below)  

## requirment 
tensorflow 1.4 or 1.5
jieba   
You need to set PYTHONPATH to be able to include ./util(for using gezi and melt) ./other/slim(I use slim but modified a bit and copy to dir other)    

## prepare data    
first turn train data to tfrecords with below filds  
image_name   
image_data(if raw image bytes, used for finetune image model also can used before finetune) or image_feature(if pre calc image feature as float vector)  
text(caption words(jieba segmented) ids)  
text_str(caption string)  

## more info
For single model (InceptionV4 + Luong attention) it can achive cider 1.5 without finetune image model and 1.78 after finetune, using label smoothing it can improve to 1.81.    
If using nasnet cant got cider 1.75 before finetune and 1.80 after finetune. Notice nasnet might be slow if using 1 gpu, you'd better set 2 gpu with each gpu batch size 8 for gtx1080ti or each gpu batch size 16 for p40.  
Support scst reinforcement learning but not improve you can help to contribute.  
If you want to fast train you can pre dump image features ( say 64 image features for InceptionV4 model) for each image,  by doing this you can achive one epoch per hour roughly and 1 day maybe enough to got cider 1.5. For finetune you need to use image model during traning that will be slow 7.5 hour per epoch, it might need 1 week to get to 1.78 then you can set label smoothng to 0.1 to finetune 3 epochs (using 1day) to get 1.81.  
Show attention tell can got good result after ensemble and gbdt rerank. But if you want to get better performance using single model  you might need to switch to "bottom up and top down attention for image captioning and visual question answering"  
