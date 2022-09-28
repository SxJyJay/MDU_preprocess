from nusc_image_projection import read_file, to_batch_tensor, to_tensor, projectionV2, reverse_view_points, get_obj
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from model import uncertainty_net

import numpy as np
from nusc_image_projection import read_file, to_batch_tensor, to_tensor, projectionV2, reverse_view_points, get_obj
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

import matplotlib.pyplot as plt
# from ip_basic import depth_map_utils
cmap = plt.cm.jet

import cv2
import os
from PIL import Image
import torchvision.transforms as transforms
import ipdb        
from tqdm import tqdm

def init_detector(args):
    from CenterNet2.train_net import setup
    from detectron2.engine import DefaultPredictor
    
    cfg = setup(args)
    predictor = DefaultPredictor(cfg)
    return predictor 

class PaintDataSet(Dataset):
    def __init__(
        self,
        info_path,
        instance_segmentor
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
        self.segmentor = instance_segmentor

    @torch.no_grad()
    def __getitem__(self, index):
        info = self.sweeps[index]
        tokens = info['lidar_path'].split('/')
        # output_path = os.path.join(*tokens[:-2], tokens[-2]+"_VIRTUAL", tokens[-1]+'.pkl.npy')
        # if os.path.isfile(output_path):
        #     return [] 

        all_cams_path = info['all_cams_path']

        all_data = [info]
        for path in all_cams_path:
            # original_image = cv2.imread(path)
            # # convert from BGR to RGB
            # original_image = original_image[:, :, ::-1]
            
            # # if self.predictor.input_format == "RGB":
            # #     # whether the model expects BGR inputs or RGB
            # #     original_image = original_image[:, :, ::-1]
            # height, width = original_image.shape[:2]
            # # image = self.predictor.aug.get_transform(original_image).apply_image(original_image)
            # image = original_image
            # image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            
            # inputs = {"image": image, "height": height, "width": width}

            original_image = Image.open(path)
            image = np.array(original_image)
            height, width = image.shape[:2]
            image = self.segmentor.aug.get_transform(image).apply_image(image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            
            inputs = {"original_image": original_image, "image": image, "height": height, "width": width}
            
            all_data.append(inputs) 

        return all_data 
    
    def __len__(self):
        return len(self.sweeps)

def simple_collate(batch_list):
    assert len(batch_list)==1
    batch_list = batch_list[0]
    return batch_list

def save_depth_as_uint16png_upload(img, filename):
    #from tensor
    img = np.squeeze(img.data.cpu().numpy())
    img = (img * 256.0).astype('uint16')
    img_buffer = img.tobytes()
    imgsave = Image.new("I", img.T.shape)
    imgsave.frombytes(img_buffer, 'raw', "I;16")
    imgsave.save(filename)

def save_depth_as_uint8colored(img, filename):
    #from tensor
    if isinstance(img, torch.Tensor):
        img = np.squeeze(img.data.cpu().numpy())
    img = depth_colorize(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)

def depth_colorize(depth):
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
    return depth.astype('uint8')

def is_within_mask(points_xyc, masks, H=900, W=1600):
    seg_mask = masks[:, :-1].reshape(-1, W, H) # (num_ins, W, H)
    camera_id = masks[:, -1] # (num_ins, cam_id)
    points_xyc = points_xyc.long()
    valid = seg_mask[:, points_xyc[:, 0], points_xyc[:, 1]] * (camera_id[:, None] == points_xyc[:, -1][None])
    return valid.transpose(1, 0) 

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
    # masks = masks.reshape(-1, 900, 1600).permute(0, 2, 1).reshape(-1, 1600*900)
    masks = masks.reshape(-1, 900, 1600)
    # ipdb.set_trace()
    return labels, scores, masks

@torch.no_grad()
def generate_instance_masks(segmentor, data, num_camera=6):
    one_hot_labels = [] 
    for i in range(10):
        one_hot_label = torch.zeros(10, device='cuda:0', dtype=torch.float32)
        one_hot_label[i] = 1
        one_hot_labels.append(one_hot_label)

    one_hot_labels = torch.stack(one_hot_labels, dim=0) 

    masks = [] 
    labels = []

    result = segmentor.model(data[1:])

    for cam_id in range(num_camera):
        pred_label, score, pred_mask = postprocess(result[cam_id])
        transformed_labels = one_hot_labels.gather(0, pred_label.reshape(-1, 1).repeat(1, 10))
        transformed_labels = torch.cat([transformed_labels, score.unsqueeze(-1)], dim=1)

        masks.append(pred_mask) # pred_mask [num_ins, 900*1600 + 1(cam_id)]
        labels.append(transformed_labels) # transformed labels [num_ins, 10(one-hot) + 1(score)]

    return masks, labels # masks: 6*[pred_mask for each camera]; labels: 6*[transformed_labels for each camera]

def main(args):
    # nusc = NuScenes(version='v1.0-mini', dataroot='/share/home/jiaoyang/code/TransFusion/depth_completion/data/sets/nuscenes/', verbose=True)
    # train_scenes = splits.mini_train
    # val_scenes = splits.mini_val
    instance_segmentor = init_detector(args)

    ckpt_path = "./model_best_epoch.pth.tar"
    predictor = uncertainty_net(in_channels=4)
    
    predictor.load_state_dict(torch.load(ckpt_path)['state_dict'])
    predictor = predictor.cuda()
    predictor.eval()
    data_loader = DataLoader(
        PaintDataSet(args.info_path, instance_segmentor),
        batch_size=1,
        num_workers=6,
        collate_fn=simple_collate,
        pin_memory=True,
        shuffle=False
    )

    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader.dataset)):
        if len(data) == 0:
            continue 
        info = data[0]
        # if info['lidar_path'] in lost_items:
        #     import ipdb
        #     ipdb.set_trace()
        tokens = info['lidar_path'].split('/')
        available_root = '/share_io02_ssd/jiaoyang/nuScenes/'
        # output_path = os.path.join(*tokens[:-2], "FOREGROUND_MIXED_6NN", tokens[-1]+'.pkl.npy')
        # output_path = '/' + output_path
        # output_path = os.path.join(available_root, tokens[-3], "FOREGROUND_MIXED_1NN_10pts", tokens[-1]+'.pkl.npy')
        output_path = os.path.join(available_root, tokens[-3], "FOREGROUND_DEPTH_COMPLETION", tokens[-1]+'.pkl.npy')
        # if os.path.exists(output_path):
        #     continue

        ToTensor = transforms.ToTensor()
        
        all_cams_from_lidar = info['all_cams_from_lidar']
        all_cams_intrinsic = info['all_cams_intrinsic']
        lidar_points = read_file(info['lidar_path'])
        P = projectionV2(to_tensor(lidar_points), to_batch_tensor(all_cams_from_lidar), to_batch_tensor(all_cams_intrinsic))

        batch_input = []
        padding = (
                0, 0,
                0, 124
            )

        ###################################################################
        ############### generate depth completion results #################
        ###################################################################

        for i in range(len(P)):

            ###############################################################
            ################ generating sparse depth map ##################
            ###############################################################

            cam_path = info['all_cams_path'][i]
            P_img = P[i]
            valid = P_img[:, 3].nonzero()
            P_img_valid = P_img[valid.squeeze(1)]
            # img size: (900, 1600)
            row_coor, col_coor = P_img_valid[:, 1].long(), P_img_valid[:, 0].long()
            inds = (row_coor, col_coor)
            depth = P_img_valid[:, 2]
            sparse_depth_map = torch.zeros(900, 1600).cuda()
            sparse_depth_map = sparse_depth_map.index_put(inds, depth)

            ###############################################################
            ############### input rgb data transformation #################
            ###############################################################

            img = data[i+1]['original_image'] # Since the very first term stores meta info
            rgb_img = img
            
            # normalize input to [0, 1]
            # img = ToTensor(img).float().cuda()
            ##!! Remain to further investigate whether 0-256 depth value should be further normalized
            # normalize depth
            # sparse_depth_map = sparse_depth_map / sparse_depth_map.max()
            
            ## Do not normalize input to [0, 1]
            img = ToTensor(img).float().cuda() * 255.0

            ###############################################################
            ############### use FusionNet as the predictor ################
            ###############################################################

            input = torch.cat((sparse_depth_map.unsqueeze(0), img), 0)

            batch_input.append(input)

            # # padding = (
            # #     0, 0,
            # #     0, 124
            # # )
            # input = F.pad(input, padding)
            
            # with torch.no_grad():
            #     output = predictor(input.unsqueeze(0))[0]
            
            # # save generated depth map as uint16 for quality visualization
            # cam_path_split = cam_path.split('/')
            # output_root = os.path.join('./data/depth_completion_vis/', cam_path_split[-2])
            # os.makedirs(output_root, exist_ok=True)
            
            
            # output_path = os.path.join(output_root, cam_path_split[-1])
            # output_sp_path = os.path.join(output_root, cam_path_split[-1]+"spmap.jpg")

            # # ipdb.set_trace()
            
            # rgb_img.save(output_path)
            # origin_spmap = input[0][:900, :]
            # save_depth_as_uint8colored(origin_spmap, output_sp_path)
            # save_depth_as_uint8colored(output, output_path+"pred.jpg")

            # torch.cuda.empty_cache()

            ################################################################
            ################ use ip_basic as the predictor #################
            ################################################################
            
            # output, _ = depth_map_utils.fill_in_multiscale(sparse_depth_map.cpu().numpy(), extrapolate=True, 
            #                                             blur_type='bilateral', show_process=False)
            # output = depth_map_utils.fill_in_fast(sparse_depth_map.cpu().numpy(), extrapolate=False, 
            #                                             blur_type='gaussian')
            # cam_path_split = cam_path.split('/')
            # output_root = os.path.join('./data/depth_completion_vis_ip/', cam_path_split[-2])
            # os.makedirs(output_root, exist_ok=True)
            
            # output_path = os.path.join(output_root, cam_path_split[-1])

            # ipdb.set_trace()
            # save_depth_as_uint8colored(output, output_path+"pred.jpg")
        
        batch_input = torch.stack(batch_input, 0)
        # To be compatible with KITTI input size
        batch_input = F.pad(batch_input, padding)
        
        with torch.no_grad():
            batch_output = predictor(batch_input)[0]

        # Restore nuscenes' original size
        batch_output = batch_output[:, :, :900, :]
        # batch_output = batch_output[:,:,:900,:].cpu().numpy()
        # batch_output = batch_output * 256.0
        # batch_output = batch_output.astype(np.uint16)

        # for i in range(len(P)):
        #     cam_path = info['all_cams_path'][i]
        #     cam_path_split = cam_path.split('/')
            
        #     output_root = os.path.join(*cam_path_split[:-2], "DEPTH_COMPLETION", cam_path_split[-2])
        #     output_root = '/' + output_root
        #     os.makedirs(output_root, exist_ok=True)
        #     output_path = os.path.join(output_root, cam_path_split[-1]+'.pkl.npy')
            
        #     # ipdb.set_trace()
        #     # save_depth_as_uint8colored(batch_output[i], output_path+"pred.jpg")

        #     output = batch_output[i].squeeze()
        #     np.save(output_path, output)

        ###################################################################
        ############ generate instance segmentation results ###############
        ###################################################################

        masks, labels = generate_instance_masks(instance_segmentor, data)
        intrinsics = to_batch_tensor(all_cams_intrinsic)
        extrinsics = to_batch_tensor(all_cams_from_lidar)

        fg_pxls_all_cams, fg_pts_all_cams = [], []
        for i in range(len(masks)):
            masks_cur_cam = masks[i]
            instance_num = masks_cur_cam.shape[0]
            if instance_num == 0:
                fg_pxls_all_cams.append(np.zeros((0, 15)))
                fg_pts_all_cams.append(np.zeros((0,3)))
                continue

            depth_cur_cam = batch_output[i].squeeze()
            labels_cur_cam = labels[i]

            fg_pxl_instances = masks_cur_cam.nonzero() # [instance_id, row_id, col_id]
            row_ids, col_ids = fg_pxl_instances[:, 1], fg_pxl_instances[:, 2] # shape: (instance_num)
            instance_ids = fg_pxl_instances[:, 0] # shape: (instance_num)

            fg_pxl_labels = labels_cur_cam[instance_ids] # shape: (instance_num)
            fg_pxl_depths = depth_cur_cam[row_ids, col_ids] # shape: (instance_num)

            ## when unproject to 3D space, need to transpose row_id and col_id in the camera world !!!
            # fg_pxl_cur_cam = torch.stack([row_ids, col_ids, fg_pxl_depths], dim=1)
            fg_pxl_cur_cam = torch.stack([col_ids, row_ids, fg_pxl_depths, instance_ids], dim=1)
            fg_pxl_cur_cam = torch.cat([fg_pxl_cur_cam, fg_pxl_labels], dim=1) # shape: (instance_num, 2(u,v) + 1(d) + 1(instance_id) + 11(label))

            ## reverse mapping 2D points to 3D world
            fg_pxl_cur_cam_padded = torch.cat(
                [fg_pxl_cur_cam[:,:2].transpose(1,0).float(),
                torch.ones((1, len(fg_pxl_cur_cam)), device=fg_pxl_cur_cam.device, dtype=torch.float32)],
                dim=0
            )
            
            fg_pts_cur_cam = reverse_view_points(fg_pxl_cur_cam_padded, fg_pxl_cur_cam[:,2], intrinsics[i])
            fg_pts_cur_cam[:3] = torch.matmul(torch.inverse(extrinsics[i]),
                        torch.cat([
                            fg_pts_cur_cam[:3, :],
                            torch.ones((1, fg_pts_cur_cam.shape[1]), dtype=torch.float32, device=fg_pts_cur_cam.device)
                        ], dim=0)
            )[:3]
            
            fg_pts_cur_cam = fg_pts_cur_cam.transpose(1,0) # shape: (instance_num, 3(x,y,z))

            # save at most 1w pts for each instance
            indices = len(fg_pxl_cur_cam)
            if indices > 10000:
                selected_indices = torch.randperm(indices, device=fg_pxl_cur_cam.device)[:10000]
                fg_pxl_cur_cam = fg_pxl_cur_cam[selected_indices]
                fg_pts_cur_cam = fg_pts_cur_cam[selected_indices]
            
            fg_pxls_all_cams.append(fg_pxl_cur_cam.cpu().numpy())
            fg_pts_all_cams.append(fg_pts_cur_cam.cpu().numpy())

        # ## generate real points and their ids    
        # camera_ids = torch.arange(6, dtype=torch.float32, device='cuda:0').reshape(6, 1, 1).repeat(1, P.shape[1], 1)
        # points_xyc = torch.cat([P, camera_ids], dim=-1)
        # points_xyc = points_xyc.reshape(-1, 5)[:, [0,1,4]]
        # # append camera id after per camera mask
        # masks_c = []
        # for i in range(len(masks)):
        #     cur_mask = masks[i].reshape(-1, 900*1600)
        #     cam_id = torch.tensor(i, dtype=torch.float32, device='cuda:0').reshape(1,1).repeat(cur_mask.shape[0], 1)
        #     mask_c = torch.cat([cur_mask, cam_id], dim=1)
        #     masks_c.append(mask_c)
        # masks_c = torch.cat(masks_c, dim=0)
        

        # valid = is_within_mask(points_xyc, masks_c)
        # valid = valid.reshape(6, -1, valid.shape[-1]) # shape: (6, num_pts, instance_num)
        
        real_fg_pxls_all_cams, real_fg_pts_all_cams = [], []
        for i in range(len(labels)):
            
            instance_mask = masks[i]
            if len(instance_mask) == 0:
                real_fg_pxls_all_cams.append(np.zeros((0,15)))
                real_fg_pts_all_cams.append(np.zeros((0,3)))
                continue

            labels_cur_cam = labels[i]
            P_img = P[i]
            valid_pts_mask = P_img[:, 3].nonzero()

            # filter points outside the image plane
            P_img_valid = P_img[valid_pts_mask.squeeze(1)]
            lidar_points_valid = to_tensor(lidar_points)[valid_pts_mask.squeeze(1)]
            
            # filter points outside the instance mask
            valid_instance_mask = instance_mask[:,P_img_valid[:,1].long(),P_img_valid[:,0].long()].t()
            valid_fg_mask = valid_instance_mask.sum(1)
            valid_fg_mask = valid_fg_mask.nonzero().squeeze(1)
            valid_fg_pxl = P_img_valid[valid_fg_mask]
            valid_fg_pts = lidar_points_valid[valid_fg_mask]
            
            valid_fg_instance_ids = torch.argmax(valid_instance_mask.float(), dim=1)[valid_fg_mask]
            valid_fg_labels = labels_cur_cam[valid_fg_instance_ids]

            real_fg_pxl_per_cam = torch.cat([valid_fg_pxl[:,:3], valid_fg_instance_ids.unsqueeze(1), valid_fg_labels], dim=1) # shape: (instance_num, 2(u,v) + 1(d) + 1(instance_id) + 11(label))
            real_fg_pts_per_cam = valid_fg_pts[:,:3]

            real_fg_pxls_all_cams.append(real_fg_pxl_per_cam.cpu().numpy())
            real_fg_pts_all_cams.append(real_fg_pts_per_cam.cpu().numpy())

        # save results
        
        data_dict = {
            'virtual_pixel_indices': fg_pxls_all_cams,
            'real_pixel_indices': real_fg_pxls_all_cams,
            'virtual_points': fg_pts_all_cams,
            'real_points': real_fg_pts_all_cams
        }
       
        np.save(output_path, data_dict)

        torch.cuda.empty_cache()

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

    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if not os.path.isdir('/share_io02_ssd/jiaoyang/nuScenes/samples/FOREGROUND_DEPTH_COMPLETION'):
        os.mkdir('/share_io02_ssd/jiaoyang/nuScenes/samples/FOREGROUND_DEPTH_COMPLETION')

    if not os.path.isdir('/share_io02_ssd/jiaoyang/nuScenes/sweeps/FOREGROUND_DEPTH_COMPLETION'):
        os.mkdir('/share_io02_ssd/jiaoyang/nuScenes/sweeps/FOREGROUND_DEPTH_COMPLETION')

    main(args)
