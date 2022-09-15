pwd
ls
pwd
nvidia-smi
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 29501 --nproc_per_node=2 \
    main.py \
    -m dab_deformable_detr \
    --transformer_activation relu \
    --backbone swin_tiny \
    --output_dir /openseg_blob_new/v-yiduohao/outputs-aml/playground \
    --batch_size 2 \
    --epochs 12 \
    --lr_drop 11 \
    --coco_path /openseg_blob_new/dataset/coco \
    --weight_decay_backbone 0.05
