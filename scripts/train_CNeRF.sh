# multi GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 1234 --nproc_per_node 4 train_CNeRF.py --batch 1 --chunk 1 --expname CNeRF --dataset_path /your_lmdb_dataset_path --size 64

# single GPU
# python train_CNeRF.py --batch 4 --chunk 2 --expname CNeRF --dataset_path /your_lmdb_dataset_path --size 64