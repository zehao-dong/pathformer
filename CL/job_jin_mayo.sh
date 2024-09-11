#BSUB -o gin_mayo.%J
#BSUB -N
#BSUB -J ginmayoGPUjob
#BSUB -gpu 'num=1:gmodel=NVIDIAGeForceGTX1080Ti'
cd /project/tantra/zehao/Research/pathformer
python run_gin.py --dataset mayo --epochs 80 --n_nodes 3000
python run_gin.py --dataset mayo --epochs 80 --n_nodes 3000 --use_cl --readout
python run_gin.py --dataset mayo --epochs 80 --n_nodes 3000 --use_cl 
