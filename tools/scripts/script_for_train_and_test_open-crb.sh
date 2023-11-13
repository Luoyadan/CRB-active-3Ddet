# KITTI
CFG=cfgs/owal-3d_kitti_models_unk_ped/secondiou_owal-3d_open-crb.yaml
python train.py --cfg_file $CFG --batch_size 8 && \
python test.py  --cfg_file $CFG --batch_size 8 --eval_all && \
python visualize.py  --cfg_file $CFG --batch_size 8 --eval_all

# nuScenes
#CFG=cfgs/owal-3d_nuscenes_models/secondiou_multihead_owal-3d_open-crb.yaml
#python train.py --cfg_file $CFG --batch_size 8 && \
#python test.py  --cfg_file $CFG --batch_size 8 --eval_all && \
#python visualize.py  --cfg_file $CFG --batch_size 8 --eval_all