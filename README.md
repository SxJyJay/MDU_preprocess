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
Don't forget to modify the path of saving the created data.
