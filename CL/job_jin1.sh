#BSUB -o gin_ros.%J
#BSUB -N
#BSUB -J ginrosGPUjob
#BSUB -gpu 'num=1:gmodel=NVIDIAA40'
cd /project/tantra/zehao/Research/pathformer
python run_gin.py --dataset mayo --epochs 80 --n_nodes 3000
python run_gin.py --dataset rosmap --epochs 80 --n_nodes 3000


gcn: mayo f1_score: 0.5054469380297866 percision 0.5094062316284539 recall 0.5079365079365079 Test accuracy: 0.53125 test loss 7607.108688354492
gcn rosmap Test accuracy: 0.4916666666666667 test loss 1639.214966668023 f1_score: 0.5725348963114658 percision 0.6078850768045595 recall 0.5518207282913166
gin mayo: Test accuracy: 0.46875 test loss 5513.488815307617 0.46424178133191896 percision 0.4698286061291369 recall 0.46835443037974683
gin rosmap Test accuracy: 0.2916666666666667 test loss 3505.8212890625 f1_score: 0.592733127366085 percision 0.6350242929190297 recall 0.5714285714285714



gcn_uno: mayo Test accuracy: 0.5625 test loss 0.7275327835232019 f1_score: 0.6232377874392417 percision 0.6285243569889577 recall 0.6265822784810127
gin_uno: mayo Test accuracy: 0.59375 test loss 0.8600013758987188 f1_score: 0.7331490529279163 percision 0.7359251678384957 recall 0.7341772151898734

gcn_uno: rosmap: 1_score: 0.6491895701198026 percision 0.6773333333333333 recall 0.7333333333333333 Test accuracy: 0.6944444444444444 test loss 0.5887549324995942
gin_uno: rosmap: f1_score: 0.6585553370284284 percision 0.6650853100291303 recall 0.7263157894736842 Test accuracy: 0.6805555555555556 test loss 0.602389613373412


gin_cl mayo: f1_score: 0.5394503492864148 percision 0.5394901394901395 recall 0.5396825396825397 Test accuracy: 0.53125 test loss 193.70205688476562




