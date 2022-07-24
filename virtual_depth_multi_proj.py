from nusc_image_projection import read_file, to_batch_tensor, to_tensor, projectionV2, reverse_view_points, get_obj
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm 
import argparse 
import numpy as np 
import torch 
import cv2 
import os
import ipdb 
H=900
W=1600

# os.environ["CUDA_LAUNCH_BLOCKING"]='1'

class PaintDataSet(Dataset):
    def __init__(
        self,
        info_path,
        predictor
    ):
        infos = get_obj(info_path)
        sweeps = []

        paths = set()

        for info in infos:
            
            if info['lidar_path'] not in paths:
                paths.add(info['lidar_path'])
                sweeps.append(info)

            for sweep in info['sweeps']:
                if sweep['lidar_path'] not in paths: 
                    sweeps.append(sweep)
                    paths.add(sweep['lidar_path'])

        self.sweeps = sweeps
        self.predictor = predictor

    @torch.no_grad()
    def __getitem__(self, index):
        info = self.sweeps[index]
        tokens = info['lidar_path'].split('/')
        output_path = os.path.join(*tokens[:-2], tokens[-2]+"_VIRTUAL", tokens[-1]+'.pkl.npy')
        if os.path.isfile(output_path):
            return [] 

        all_cams_path = info['all_cams_path']

        all_data = [info]
        for path in all_cams_path:
            original_image = cv2.imread(path)
            
            if self.predictor.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.predictor.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            
            inputs = {"image": image, "height": height, "width": width}
            
            all_data.append(inputs) 

        return all_data 
    
    def __len__(self):
        return len(self.sweeps)

def is_within_mask(points_xyc, masks, H=900, W=1600):
    seg_mask = masks[:, :-1].reshape(-1, W, H) # (num_ins, W, H)
    camera_id = masks[:, -1] # (num_ins, cam_id)
    points_xyc = points_xyc.long()
    valid = seg_mask[:, points_xyc[:, 0], points_xyc[:, 1]] * (camera_id[:, None] == points_xyc[:, -1][None])
    return valid.transpose(1, 0) 

@torch.no_grad()
def add_virtual_mask(masks, labels, points, raw_points, num_virtual=50, dist_thresh=3000, num_camera=6, intrinsics=None, transforms=None, K=6):
    # points: [cam_num, num_pts, 3(x,y,depth)+1(valid)+1(cam_id)]
    # raw_points: [num_pts, 3(x,y,z)+1(valid)]
    # masks: [num_ins, 900*1600 + 1(cam_id)]
    # labels: [num_ins, 10(one-hot) + 1(score)]
    points_xyc = points.reshape(-1, 5)[:, [0, 1, 4]] # x, y, z, valid_indicator, camera id 


    valid = is_within_mask(points_xyc, masks) # (num_pts, num_ins)
    # jointly consider points' original valid indicator and whether falling inside foreground mask indicator
    valid = valid * points.reshape(-1, 5)[:, 3:4] # num_points, num_instances, each row indicates the belonging of the current points to all instances

    # remove camera id from masks 
    camera_ids = masks[:, -1]
    masks = masks[:, :-1]

    box_to_label_mapping = torch.argmax(valid.float(), dim=1).reshape(-1, 1).repeat(1, 11)
    point_labels = labels.gather(0, box_to_label_mapping)
    # to also include background points, manually add a new label of "background" for points falling outside instance masks
    point_labels *= (valid.sum(dim=1, keepdim=True) > 0 )  # mop point label info of invalid points, as argmax(<all zeros>) will return 0

    foreground_real_point_mask = (valid.sum(dim=1, keepdim=True) > 0).reshape(num_camera, -1).sum(dim=0).bool() # num_points*num_camera, num_instance --> num_points*num_camera --> num_points  

    offsets = [] 
    for mask in masks:
        indices = mask.reshape(W, H).nonzero()
        selected_indices = torch.randperm(len(indices), device=masks.device)[:num_virtual]
        if len(selected_indices) < num_virtual:
                selected_indices = torch.cat([selected_indices, selected_indices[
                    selected_indices.new_zeros(num_virtual-len(selected_indices))]])

        offset = indices[selected_indices]
        offsets.append(offset)
    
    offsets = torch.stack(offsets, dim=0) # [num_ins, num_vir, 2] virtual points randomly picked up from instance masks
    virtual_point_instance_ids = torch.arange(1, 1+masks.shape[0], 
        dtype=torch.float32, device='cuda:0').reshape(masks.shape[0], 1, 1).repeat(1, num_virtual, 1)

    # append instance id after offsets
    virtual_points = torch.cat([offsets, virtual_point_instance_ids], dim=-1).reshape(-1, 3) # [num_ins*num_vir, 3], append instance ids 
    virtual_point_camera_ids = camera_ids.reshape(-1, 1, 1).repeat(1, num_virtual, 1).reshape(-1, 1) # [num_ins*num_vir, 1]

    # append instance id after foreground reference points
    valid_mask = valid.sum(dim=1)>0
    real_point_instance_ids = (torch.argmax(valid.float(), dim=1) + 1)[valid_mask]
    real_points = torch.cat([points_xyc[:, :2][valid_mask], real_point_instance_ids[..., None]], dim=-1) # [num_ref, 3], append instance ids

    # avoid matching across instances 
    real_points[:, -1] *= 1e4 
    virtual_points[:, -1] *= 1e4 

    if len(real_points) == 0:
        return None 
    # ipdb.set_trace()
    dist = torch.norm(virtual_points.unsqueeze(1) - real_points.unsqueeze(0), dim=-1)  # [num_ins*num_vir, num_ref]
    
    # select K nearest reference points
    # if reference points number is small than predefined K, then update K value
    if dist.shape[1] < K:
        K = dist.shape[1]
    K_nearest_dist, K_nearest_indices = torch.topk(dist, K, dim=1, sorted=True, largest=False) # [num_ins*num_vir, K]
    mask = K_nearest_dist < dist_thresh

    # find the indices of reference points in original projected points
    indices = valid_mask.nonzero(as_tuple=False).reshape(-1)
    K_nearest_indices = indices[K_nearest_indices]
    K_nearest_xy = points.reshape(-1, 5)[K_nearest_indices.reshape(-1), 0:2].reshape(-1, K, 2)
    K_nearest_depths = points.reshape(-1, 5)[K_nearest_indices.reshape(-1), 2].reshape(-1, K)
    K_nearest_labels = point_labels[K_nearest_indices.reshape(-1), :].reshape(-1, K, 11)
    
    virtual_points = virtual_points.unsqueeze(1).repeat(1, K, 1)
    raw_points_cams = raw_points.unsqueeze(0).repeat(6, 1, 1)
    cam_points_valid = valid.reshape(points.shape[0], points.shape[1], -1)
    cam_points_labels = point_labels.reshape(points.shape[0], points.shape[1], -1)

    all_virtual_pixel_indices = []
    all_virtual_points = []
    all_virtual_point_labels = []
    all_real_pixel_indices = [] 
    all_real_points = [] 
    all_real_point_labels = []

    for i in range(num_camera):
        
        # processing virtual points (instance-level)
        camera_mask = (virtual_point_camera_ids == i).squeeze()
        per_cam_virtual_pts = virtual_points[camera_mask]
        per_cam_K_nearest_dist = K_nearest_dist[camera_mask]
        
        per_cam_K_nearest_xy = K_nearest_xy[camera_mask]
        per_cam_K_nearest_depths = K_nearest_depths[camera_mask]
        per_cam_K_nearest_labels = K_nearest_labels[camera_mask]
        per_cam_mask = mask[camera_mask]

        # make a virtual pts id vec
        per_cam_virtual_pts_ids = torch.arange(1, 1+per_cam_mask.shape[0], dtype=torch.float32, device='cuda:0').reshape(per_cam_mask.shape[0], 1).repeat(1, K)

        # select valid reference points infos
        per_cam_virtual_pts = per_cam_virtual_pts[per_cam_mask]
        per_cam_K_nearest_dist = per_cam_K_nearest_dist[per_cam_mask]
        per_cam_K_nearest_xy = per_cam_K_nearest_xy[per_cam_mask]
        per_cam_K_nearest_depths = per_cam_K_nearest_depths[per_cam_mask]
        per_cam_K_nearest_labels = per_cam_K_nearest_labels[per_cam_mask]
        per_cam_virtual_pts_ids = per_cam_virtual_pts_ids[per_cam_mask]

        # compute real world 3d points
        per_cam_virtual_pts_padded = torch.cat(
            [per_cam_virtual_pts[:, :2].transpose(1, 0).float(),
            torch.ones((1, len(per_cam_virtual_pts)), device=per_cam_virtual_pts.device, dtype=torch.float32)],
            dim=0
        )
        per_cam_virtual_pts_3d = reverse_view_points(per_cam_virtual_pts_padded, per_cam_K_nearest_depths, intrinsics[i])
        per_cam_virtual_pts_3d[:3] = torch.matmul(torch.inverse(transforms[i]),
                    torch.cat([
                        per_cam_virtual_pts_3d[:3, :],
                        torch.ones((1, per_cam_virtual_pts_3d.shape[1]), dtype=torch.float32, device=per_cam_virtual_pts_3d.device)
                    ], dim=0)
            )[:3]
        per_cam_virtual_pts_3d = per_cam_virtual_pts_3d.transpose(1,0)
        
        per_cam_all_virtual_pixel_indices = per_cam_virtual_pts[:,:2]
        per_cam_all_virtual_pts = per_cam_virtual_pts_3d
        per_cam_all_virtual_labels = per_cam_K_nearest_labels

        # # collect reference points infos according to virtual_pts_ids, since the index results of mask with shape of [num_ins, K] is a 1-dimension vector
        # unique_virtual_pts_indices = torch.unique(per_cam_virtual_pts_ids)

        # per_cam_all_virtual_pixel_indices, per_cam_all_virtual_pts = [], []
        # per_cam_all_virtual_labels = []

        # for cur_virtual_pts_id in unique_virtual_pts_indices:
            
        #     cur_virtual_pts_mask = (per_cam_virtual_pts_ids == cur_virtual_pts_id)
        #     per_cam_cur_virtual_pts = per_cam_virtual_pts[cur_virtual_pts_mask]
        #     per_cam_cur_K_nearest_xy = per_cam_K_nearest_xy[cur_virtual_pts_mask]
        #     per_cam_cur_K_nearest_depths = per_cam_K_nearest_depths[cur_virtual_pts_mask]
        #     per_cam_cur_K_nearest_labels = per_cam_K_nearest_labels[cur_virtual_pts_mask]
            
        #     # K-nn depths copy (by: jiezq)
        #     per_cam_cur_virtual_pts_3d = per_cam_virtual_pts_3d[cur_virtual_pts_mask]

        #     per_cam_all_virtual_pixel_indices.append(per_cam_cur_virtual_pts)
        #     per_cam_all_virtual_pts.append(per_cam_cur_virtual_pts_3d)
        #     per_cam_all_virtual_labels.append(per_cam_cur_K_nearest_labels)

        # if len(unique_virtual_pts_indices) != 0:
        #     per_cam_all_virtual_pixel_indices = torch.cat(per_cam_all_virtual_pixel_indices, dim=0)
        #     per_cam_all_virtual_pts = torch.cat(per_cam_all_virtual_pts, dim=0)
        #     per_cam_all_virtual_labels = torch.cat(per_cam_all_virtual_labels, dim=0)
        # else:
        #     per_cam_all_virtual_pixel_indices = per_cam_cur_virtual_pts
        #     per_cam_all_virtual_pts = per_cam_cur_virtual_pts_3d
        #     per_cam_all_virtual_labels = per_cam_cur_K_nearest_labels
        
        all_virtual_pixel_indices.append(per_cam_all_virtual_pixel_indices)
        # append labels after points
        all_virtual_points.append(torch.cat([per_cam_all_virtual_pts, per_cam_all_virtual_labels], dim=-1))
        # all_virtual_points.append(per_cam_all_virtual_pts)
        # all_virtual_point_labels.append(per_cam_all_virtual_labels)
        
        # processing real points (point level)
        per_cam_points_valid = cam_points_valid[i].sum(1).bool()
        per_cam_points = points[i]
        per_cam_raw_points = raw_points_cams[i]
        per_cam_point_labels = cam_points_labels[i]
        all_real_pixel_indices.append(per_cam_points[per_cam_points_valid][:, :2])
        # append labels after points
        all_real_points.append(torch.cat(
            [per_cam_raw_points[per_cam_points_valid][:, :3], per_cam_point_labels[per_cam_points_valid]], dim=-1))
        # all_real_points.append(per_cam_raw_points[per_cam_points_valid][:, :3])
        # all_real_point_labels.append(per_cam_point_labels[per_cam_points_valid])

    return all_virtual_pixel_indices, all_real_pixel_indices, all_virtual_points, all_real_points

def init_detector(args):
    from CenterNet2.train_net import setup
    from detectron2.engine import DefaultPredictor
    
    cfg = setup(args)
    predictor = DefaultPredictor(cfg)
    return predictor 

def postprocess(res):
    result = res['instances']
    labels = result.pred_classes
    scores = result.scores 
    masks = result.pred_masks.reshape(scores.shape[0], 1600*900) 
    boxes = result.pred_boxes.tensor

    # remove empty mask and their scores / labels 
    empty_mask = masks.sum(dim=1) == 0

    labels = labels[~empty_mask]
    scores = scores[~empty_mask]
    masks = masks[~empty_mask]
    boxes = boxes[~empty_mask]
    masks = masks.reshape(-1, 900, 1600).permute(0, 2, 1).reshape(-1, 1600*900)
    # ipdb.set_trace()
    return labels, scores, masks


@torch.no_grad()
def process_one_frame(info, predictor, data, num_camera=6):
    all_cams_from_lidar = info['all_cams_from_lidar']
    all_cams_intrinsic = info['all_cams_intrinsic']
    lidar_points = read_file(info['lidar_path'])

    one_hot_labels = [] 
    for i in range(10):
        one_hot_label = torch.zeros(10, device='cuda:0', dtype=torch.float32)
        one_hot_label[i] = 1
        one_hot_labels.append(one_hot_label)

    one_hot_labels = torch.stack(one_hot_labels, dim=0) 

    masks = [] 
    labels = [] 
    camera_ids = torch.arange(6, dtype=torch.float32, device='cuda:0').reshape(6, 1, 1)

    result = predictor.model(data[1:])

    for camera_id in range(num_camera):
        pred_label, score, pred_mask = postprocess(result[camera_id])
        camera_id = torch.tensor(camera_id, dtype=torch.float32, device='cuda:0').reshape(1,1).repeat(pred_mask.shape[0], 1)
        pred_mask = torch.cat([pred_mask, camera_id], dim=1)
        transformed_labels = one_hot_labels.gather(0, pred_label.reshape(-1, 1).repeat(1, 10))
        transformed_labels = torch.cat([transformed_labels, score.unsqueeze(-1)], dim=1)

        masks.append(pred_mask) # pred_mask [num_ins, 900*1600 + 1(cam_id)]
        labels.append(transformed_labels) # transformed labels [num_ins, 10(one-hot) + 1(score)]
    
    masks = torch.cat(masks, dim=0)
    labels = torch.cat(labels, dim=0)

    # mapping 3D points to 2D camera images
    P = projectionV2(to_tensor(lidar_points), to_batch_tensor(all_cams_from_lidar), to_batch_tensor(all_cams_intrinsic))
    camera_ids = torch.arange(6, dtype=torch.float32, device='cuda:0').reshape(6, 1, 1).repeat(1, P.shape[1], 1)
    P = torch.cat([P, camera_ids], dim=-1) # P [cam_num, num_pts, 3(x,y,depth)+1(valid)+1(cam_id)]

    if len(masks) == 0:
        res = None
    else:
        res  = add_virtual_mask(masks.clone(), labels.clone(), P.clone(), to_tensor(lidar_points), num_virtual=200,
            intrinsics=to_batch_tensor(all_cams_intrinsic), transforms=to_batch_tensor(all_cams_from_lidar), K=6)

    
    if res is not None:
        virtual_pixel_indices, real_pixel_indices, virtual_points, real_points = res
        # import ipdb
        # ipdb.set_trace()
        num_camera = len(virtual_pixel_indices)
        for cam_id in range(num_camera):
            virtual_pixel_indices[cam_id] = virtual_pixel_indices[cam_id].cpu().numpy()
            real_pixel_indices[cam_id] = real_pixel_indices[cam_id].cpu().numpy()
            virtual_points[cam_id] = virtual_points[cam_id].cpu().numpy()
            real_points[cam_id] = real_points[cam_id].cpu().numpy()      

        return virtual_pixel_indices, real_pixel_indices, virtual_points, real_points
    else:
        return None 


def simple_collate(batch_list):
    assert len(batch_list)==1
    batch_list = batch_list[0]
    return batch_list


def main(args):
    predictor = init_detector(args)
    data_loader = DataLoader(
        PaintDataSet(args.info_path, predictor),
        batch_size=1,
        num_workers=8,
        collate_fn=simple_collate,
        pin_memory=True,
        shuffle=False
    )
    
    lost_items = [
        './data/nuscenes/sweeps/LIDAR_TOP/n008-2018-08-29-16-04-13-0400__LIDAR_TOP__1535573502949680.pcd.bin',
        './data/nuscenes/sweeps/LIDAR_TOP/n015-2018-08-03-15-31-50+0800__LIDAR_TOP__1533281735799151.pcd.bin',
        './data/nuscenes/sweeps/LIDAR_TOP/n015-2018-11-21-19-11-29+0800__LIDAR_TOP__1542798706447763.pcd.bin'
    ]

    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader.dataset)):
        if len(data) == 0:
            continue 
        info = data[0]
        # if info['lidar_path'] in lost_items:
        #     import ipdb
        #     ipdb.set_trace()
        tokens = info['lidar_path'].split('/')
        available_root = '/share_io02_hdd/jiaoyang/nuScenes/'
        # output_path = os.path.join(*tokens[:-2], "FOREGROUND_MIXED_6NN", tokens[-1]+'.pkl.npy')
        # output_path = '/' + output_path
        output_path = os.path.join(available_root, tokens[-3], "FOREGROUND_MIXED_6NN_200pts", tokens[-1]+'.pkl.npy')
        
        if os.path.exists(output_path):
            continue

        res = process_one_frame(info, predictor, data)

        if res is not None:
            virtual_pixel_indices, real_pixel_indices, virtual_points, real_points = res 
        else:
            virtual_pixel_indices = []
            real_pixel_indices = []
            virtual_points = []
            real_points = []

        data_dict = {
            'virtual_pixel_indices': virtual_pixel_indices, 
            'real_pixel_indices': real_pixel_indices,
            'virtual_points': virtual_points,
            'real_points': real_points
        }

        # import ipdb
        # ipdb.set_trace()

        np.save(output_path, data_dict)
        # torch.cuda.empty_cache() if you get OOM error 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="CenterPoint")
    parser.add_argument('--info_path', type=str, required=True)
    parser.add_argument('--config-file', type=str, default='c2_config/nuImages_CenterNet2_DLA_640_8x.yaml')
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
        "See config references at "
        "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    # if not os.path.isdir('data/nuScenes/samples/FOREGROUND_MIXED_6NN'):
    #     os.mkdir('data/nuScenes/samples/FOREGROUND_MIXED_6NN')

    # if not os.path.isdir('data/nuScenes/sweeps/FOREGROUND_MIXED_6NN'):
    #     os.mkdir('data/nuScenes/sweeps/FOREGROUND_MIXED_6NN')

    if not os.path.isdir('/share_io02_hdd/jiaoyang/nuScenes/samples/FOREGROUND_MIXED_6NN_200pts'):
        os.mkdir('/share_io02_hdd/jiaoyang/nuScenes/samples/FOREGROUND_MIXED_6NN_200pts')

    if not os.path.isdir('/share_io02_hdd/jiaoyang/nuScenes/sweeps/FOREGROUND_MIXED_6NN_200pts'):
        os.mkdir('/share_io02_hdd/jiaoyang/nuScenes/sweeps/FOREGROUND_MIXED_6NN_200pts')

    main(args)
