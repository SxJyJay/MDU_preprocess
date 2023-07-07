from nusc_image_projection import read_file, to_batch_tensor, to_tensor, projectionV2, reverse_view_points, get_obj
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm 
import argparse 
import numpy as np 
import torch 
import cv2 
import os
import json
import ipdb
import PIL.Image 
H=900
W=1600

# os.environ["CUDA_LAUNCH_BLOCKING"]='1'

def mask2polygon(mask):
    contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # mask_new, contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        contour_list = contour.flatten().tolist()
        if len(contour_list) > 4:# and cv2.contourArea(contour)>10000
            segmentation.append(contour_list)
    return segmentation

def polygons_to_mask(img_shape, polygons):
    mask = np.zeros(img_shape, dtype=np.uint8)
    all_polygons = []
    for polygon in polygons:
        polygon = np.asarray(polygon, np.int32) # 这里必须是int32，其他类型使用fillPoly会报错
        polygon=polygon.reshape(-1,2)
        all_polygons.append(polygon) # 非int32 会报错
    cv2.fillPoly(mask, all_polygons, color=1)
    return mask

# def polygons_to_mask(img_shape, polygons):
#     mask = np.zeros(img_shape, dtype=np.uint8)
#     mask = PIL.Image.fromarray(mask)
#     xy = list(map(tuple, polygons))
#     PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
#     mask = np.array(mask, dtype=bool)
#     return mask

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


def init_detector(args):
    from CenterNet2.train_net import setup
    from detectron2.engine import DefaultPredictor
    from detectron2.utils.visualizer import ColorMode, Visualizer
    
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
    # masks = masks.reshape(-1, 900, 1600).permute(0, 2, 1).reshape(-1, 1600*900)
    masks = masks.reshape(-1, 900, 1600)
    # ipdb.set_trace()
    return labels, scores, boxes, masks


@torch.no_grad()
def process_one_frame(info, predictor, data, idx, num_camera=6):
    all_cams_from_lidar = info['all_cams_from_lidar']
    all_cams_intrinsic = info['all_cams_intrinsic']
    lidar_points = read_file(info['lidar_path'])

    save_root = "./visualize/CenterNet2/" + str(idx) + "/"
    

    one_hot_labels = [] 
    for i in range(10):
        one_hot_label = torch.zeros(10, device='cuda:0', dtype=torch.float32)
        one_hot_label[i] = 1
        one_hot_labels.append(one_hot_label)

    one_hot_labels = torch.stack(one_hot_labels, dim=0) 

    masks = [] 
    labels = []
    colormap_lookuptable = [cv2.COLORMAP_JET, cv2.COLORMAP_RAINBOW, cv2.COLORMAP_HSV, cv2.COLORMAP_PARULA, cv2.COLORMAP_PLASMA,\
                            cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_CIVIDIS, cv2.COLORMAP_SPRING, cv2.COLORMAP_PINK, cv2.COLORMAP_TURBO] 
    camera_ids = torch.arange(6, dtype=torch.float32, device='cuda:0').reshape(6, 1, 1)

    result = predictor.model(data[1:])

    # for camera_id in range(num_camera):
    #     vis = Visualizer(data[1+camera_id]['image'], instance_mode=ColorMode.IMAGE)
    P = projectionV2(to_tensor(lidar_points), to_batch_tensor(all_cams_from_lidar), to_batch_tensor(all_cams_intrinsic))
    
    point_color = (0, 255, 0)
    # 线的厚度
    thickness = 2
    # 线的类型
    lineType = 4

    all_seg_infos = []

    for camera_id in range(num_camera):
        pred_label, score, pred_boxes, pred_mask = postprocess(result[camera_id])

        # # SAVE ORIGINAL IMAGE AS REFERENCE
        # image = data[1+camera_id]['image']
        # image = image.permute(1,2,0).cpu().numpy().astype(np.uint8)[:,:,::-1]

        # ori_name = "cam_" + str(camera_id) + ".png"
        # ori_save_path = os.path.join(save_root, ori_name)
        # cv2.imwrite(ori_save_path, image)

        this_cam_seg_info = []

        for ins_id in range(len(pred_label)):

            # save_name = "cam_" + str(camera_id) + "_ins_" + str(ins_id) + ".png"
            # save_path = os.path.join(save_root, save_name)

            ins_label = pred_label[ins_id]
            ins_mask = pred_mask[ins_id]
            ins_box = pred_boxes[ins_id]

            ins_mask = ins_mask.cpu().numpy().astype(np.uint8)
            poly_mask = mask2polygon(ins_mask)

            # # RECOVER BINARY MASK FOR VERIFICATION
            # if len(poly_mask) > 1:
                
            #     import ipdb
            #     ipdb.set_trace()
            #     os.makedirs(save_root, exist_ok=True)
            #     recover_ins_mask = polygons_to_mask((900, 1600), poly_mask)
            #     recover_ins_mask = recover_ins_mask * 255
            #     colored_ins_mask = cv2.applyColorMap(recover_ins_mask, colormap_lookuptable[ins_label.item()])
            #     ins_masked_image = cv2.addWeighted(image, 0.4, colored_ins_mask, 0.6, 0)
            #     cv2.imwrite(save_path, ins_masked_image)

            seg_info = {
                "ins_id": ins_id,
                "category_id": ins_label.item(),
                "bbox": ins_box.tolist(),
                "segmentation": poly_mask
            }
            this_cam_seg_info.append(seg_info)

        all_seg_infos.append(this_cam_seg_info)

    return all_seg_infos



        

    #     camera_id = torch.tensor(camera_id, dtype=torch.float32, device='cuda:0').reshape(1,1).repeat(pred_mask.shape[0], 1)
    #     pred_mask = torch.cat([pred_mask, camera_id], dim=1)
    #     transformed_labels = one_hot_labels.gather(0, pred_label.reshape(-1, 1).repeat(1, 10))
    #     transformed_labels = torch.cat([transformed_labels, score.unsqueeze(-1)], dim=1)

    #     masks.append(pred_mask) # pred_mask [num_ins, 900*1600 + 1(cam_id)]
    #     labels.append(transformed_labels) # transformed labels [num_ins, 10(one-hot) + 1(score)]
    
    # masks = torch.cat(masks, dim=0)
    # labels = torch.cat(labels, dim=0)

    # # mapping 3D points to 2D camera images
    # P = projectionV2(to_tensor(lidar_points), to_batch_tensor(all_cams_from_lidar), to_batch_tensor(all_cams_intrinsic))
    # camera_ids = torch.arange(6, dtype=torch.float32, device='cuda:0').reshape(6, 1, 1).repeat(1, P.shape[1], 1)
    # P = torch.cat([P, camera_ids], dim=-1) # P [cam_num, num_pts, 3(x,y,depth)+1(valid)+1(cam_id)]

    # if len(masks) == 0:
    #     res = None
    # else:
    #     res  = add_virtual_mask(masks.clone(), labels.clone(), P.clone(), to_tensor(lidar_points), num_virtual=50,
    #         intrinsics=to_batch_tensor(all_cams_intrinsic), transforms=to_batch_tensor(all_cams_from_lidar), K=6)

    
    # if res is not None:
    #     virtual_pixel_indices, real_pixel_indices, virtual_points, real_points = res
    #     # import ipdb
    #     # ipdb.set_trace()
    #     num_camera = len(virtual_pixel_indices)
    #     for cam_id in range(num_camera):
    #         virtual_pixel_indices[cam_id] = virtual_pixel_indices[cam_id].cpu().numpy()
    #         real_pixel_indices[cam_id] = real_pixel_indices[cam_id].cpu().numpy()
    #         virtual_points[cam_id] = virtual_points[cam_id].cpu().numpy()
    #         real_points[cam_id] = real_points[cam_id].cpu().numpy()      

    #     return virtual_pixel_indices, real_pixel_indices, virtual_points, real_points
    # else:
    #     return None 


def simple_collate(batch_list):
    assert len(batch_list)==1
    batch_list = batch_list[0]
    return batch_list


def main(args):
    predictor = init_detector(args)
    data_loader = DataLoader(
        PaintDataSet(args.info_path, predictor),
        batch_size=1,
        num_workers=6,
        collate_fn=simple_collate,
        pin_memory=True,
        shuffle=False
    )
    
    lost_items = [
        './data/nuscenes/sweeps/LIDAR_TOP/n008-2018-08-29-16-04-13-0400__LIDAR_TOP__1535573502949680.pcd.bin',
        './data/nuscenes/sweeps/LIDAR_TOP/n015-2018-08-03-15-31-50+0800__LIDAR_TOP__1533281735799151.pcd.bin',
        './data/nuscenes/sweeps/LIDAR_TOP/n015-2018-11-21-19-11-29+0800__LIDAR_TOP__1542798706447763.pcd.bin'
    ]
    debug_items = [
        './data/nuscenes/samples/LIDAR_TOP/n015-2018-09-27-15-33-17+0800__LIDAR_TOP__1538033797797687.pcd.bin'
    ]

    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader.dataset)):
        if len(data) == 0:
            continue 
        info = data[0]
        
        # if info['lidar_path'] in lost_items:
        #     import ipdb
        #     ipdb.set_trace()
        # if info['lidar_path'] not in debug_items:
        #     continue

        tokens = info['lidar_path'].split('/')
        available_root = "/home/jy/code/MVP/data/nuScenes/"
        # available_root = '/share_io02_ssd/jiaoyang/nuScenes/'
        # output_path = os.path.join(*tokens[:-2], "FOREGROUND_MIXED_6NN", tokens[-1]+'.pkl.npy')
        # output_path = '/' + output_path
        output_path = os.path.join(available_root, tokens[-3], "CAM_SEG", tokens[-1]+'.json')
        # os.makedirs(os.path.join(available_root, tokens[-3], "CAM_SEG"), exist_ok=True)
        
        # if os.path.exists(output_path):
        #     continue

        all_seg_infos = process_one_frame(info, predictor, data, idx)

        data_dict = {
            "all_seg_infos": all_seg_infos
        }

        json_str = json.dumps(data_dict)
        with open(output_path, 'w') as json_file:
            json_file.write(json_str)

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

    if not os.path.isdir('/home/jy/code/MVP/data/nuScenes/samples/CAM_SEG'):
        os.makedirs('/home/jy/code/MVP/data/nuScenes/samples/CAM_SEG', exist_ok=True)

    if not os.path.isdir('/home/jy/code/MVP/data/nuScenes/sweeps/CAM_SEG'):
        os.makedirs('/home/jy/code/MVP/data/nuScenes/sweeps/CAM_SEG', exist_ok=True)

    main(args)
