#BSUB -o gcn_ros.%J
#BSUB -N
#BSUB -J gcnrosGPUjob
#BSUB -gpu 'num=1:gmodel=NVIDIAGeForceGTX1080Ti'
cd /project/tantra/zehao/Research/pathformer
python run_gcn.py --dataset rosmap --epochs 80 --n_nodes 3000
python run_gcn.py --dataset rosmap --epochs 80 --n_nodes 3000 --use_cl --readout
python run_gcn.py --dataset rosmap --epochs 80 --n_nodes 3000 --use_cl 
