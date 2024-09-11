#BSUB -o gin_ros.%J
#BSUB -N
#BSUB -J ginrosGPUjob
#BSUB -gpu 'num=1:gmodel=NVIDIAGeForceGTX1080Ti'
cd /project/tantra/zehao/Research/pathformer
python run_gin.py --dataset rosmap --epochs 80 --n_nodes 3000
python run_gin.py --dataset rosmap --epochs 80 --n_nodes 3000 --use_cl --readout
python run_gin.py --dataset rosmap --epochs 80 --n_nodes 3000 --use_cl 
