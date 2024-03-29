# multi-depth-projection
Creating virtual 2D points via multi-depth projection

Setup steps please refer to [MVP](https://github.com/tianweiy/MVP).

Based on MVP's framework, we further devise two virtual points projection approaches, i.e., multi-depth-projection and boundary-projection. The multi-depth-projection (50 pts, 6NN) can bring over 2.0 mAP improvements on nuScenes dataset, while boundary-projection has not been experimented. To create multi-depth projection data, you can run the following command:
```
python virtual_depth_mapping_multi_proj.py --info_path data/nuScenes/infos_train_10sweeps_withvelo_filter_True.pkl  MODEL.WEIGHTS centernet2_checkpoint.pth
```
Similarly, you can also create boundary projection data by running:
```
python virtual_boundary_gen.py --info_path data/nuScenes/infos_train_10sweeps_withvelo_filter_True.pkl  MODEL.WEIGHTS centernet2_checkpoint.pth
```
You can also create depth completion-based projection data (at most 10,000 pts for each instance) by running:
```
python virtual_depth_completion_nuscenes.py --info_path data/nuScenes/infos_train_10sweeps_withvelo_filter_True.pkl  MODEL.WEIGHTS centernet2_checkpoint.pth
```
If you want to increase or reduce the number of pts per instance, please modify the upper bound of random selection [here](https://github.com/SxJyJay/multi-depth-projection/blob/e1ad4b2c3b9121edb88c9c7d65c334a3d62e3d1b/virtual_depth_completion_nuscenes.py#L398).

Don't forget to modify the path of saving the created data.

# nuscenes segmentation results generation
Generate nuscenes instance segmentation results and save them as json files.

You can use the following command to generate nuscenes segmentation results files.
```
python nusc_seg.py --info_path data/nuScenes/infos_train_10sweeps_withvelo_filter_True.pkl  MODEL.WEIGHTS centernet2_checkpoint.pth
```

Don't forget to modify the relevant [file reading](https://github.com/SxJyJay/multi-depth-projection/blob/c53b498009627af081f8c9be4ffb3fbb7e040460/nusc_seg.py#L304) and [saving](https://github.com/SxJyJay/multi-depth-projection/blob/c53b498009627af081f8c9be4ffb3fbb7e040460/nusc_seg.py#L348) path.
