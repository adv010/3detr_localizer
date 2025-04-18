import os
import numpy as np
import torch
import utils.pc_util as pc_util
from torch.utils.data import Dataset
from utils.box_util import (flip_axis_to_camera_np, flip_axis_to_camera_tensor,
                          get_3d_box_batch_np, get_3d_box_batch_tensor)
from PIL import Image
import random
import pickle
import open3d as o3d


class CustomDatasetConfig(object):
    def __init__(self):
        self.num_semcls = 1  # Update with your number of classes
        self.num_angle_bin = 12  # For orientation prediction
        self.max_num_obj = 64  # Max objects per scene
        
        # Update with your class names and IDs
        self.type2class = {
            "object": 0  # Example - add all your classes here
        }
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.class_ids = np.array([0])  # Your class IDs
        
        # For semantic segmentation (if needed)
        self.num_class_semseg = 1
        self.type2class_semseg = {"object": 0}
        self.class2type_semseg = {self.type2class_semseg[t]: t for t in self.type2class_semseg}

    def angle2class(self, angle):
        angle = angle % (2 * np.pi)
        assert angle >= 0 and angle <= 2 * np.pi
        angle_per_class = 2 * np.pi / float(self.num_angle_bin)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
        return class_id, residual_angle

    def class2angle(self, pred_cls, residual, to_label_format=True):
        angle_per_class = 2 * np.pi / float(self.num_angle_bin)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle > np.pi:
            angle = angle - 2 * np.pi
        return angle

    def class2anglebatch_tensor(self, pred_cls, residual, to_label_format=True):
        return self.class2angle(pred_cls, residual, to_label_format)

    def class2anglebatch(self, pred_cls, residual, to_label_format=True):
        return self.class2angle(pred_cls, residual, to_label_format)

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual, box_size=None):
        heading_angle = self.class2angle(heading_class, heading_residual)
        if box_size is None:
            box_size = np.array([1.0, 1.0, 1.0])  # Default size, update as needed
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle
        return obb

    def box_parametrization_to_corners(self, box_center_unnorm, box_size, box_angle):
        boxes = get_3d_box_batch_tensor(box_size, box_angle, box_center_unnorm)
        # print("BPCshape", boxes.shape)
        return boxes

    def box_parametrization_to_corners_np(self, box_center_unnorm, box_size, box_angle):
        boxes = get_3d_box_batch_np(box_size, box_angle, box_center_unnorm)
        # print("BPCNPshape", boxes.shape)
        return boxes


    @staticmethod
    def rotate_aligned_boxes(input_boxes, rot_mat):
        centers, lengths = input_boxes[:, 0:3], input_boxes[:, 3:6]
        new_centers = np.dot(centers, np.transpose(rot_mat))

        dx, dy = lengths[:, 0] / 2.0, lengths[:, 1] / 2.0
        new_x = np.zeros((dx.shape[0], 4))
        new_y = np.zeros((dx.shape[0], 4))

        for i, crnr in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
            crnrs = np.zeros((dx.shape[0], 3))
            crnrs[:, 0] = crnr[0] * dx
            crnrs[:, 1] = crnr[1] * dy
            crnrs = np.dot(crnrs, np.transpose(rot_mat))
            new_x[:, i] = crnrs[:, 0]
            new_y[:, i] = crnrs[:, 1]

        new_dx = 2.0 * np.max(new_x, 1)
        new_dy = 2.0 * np.max(new_y, 1)
        new_lengths = np.stack((new_dx, new_dy, lengths[:, 2]), axis=1)

        return np.concatenate([new_centers, new_lengths], axis=1)


class CustomDetectionDataset(Dataset):
    def __init__(
        self,
        config=None,
        split_set="train",
        root_dir=None,
        meta_data_dir=None,  # Kept for compatibility
        num_points=40000,
        all_scenes=None,
        use_color=True,
        use_height=False,
        augment=False,
        **kwargs  # Catch any additional arguments
    ):
        """
        Args:
            config: Dataset configuration object
            split_set: "train" or "val"
            root_dir: Directory containing your data files
            num_points: Number of points to sample
            use_color: Whether to use RGB colors
            use_height: Whether to use height features
            augment: Whether to apply augmentation
        """
        # Initialize base class
        super().__init__()
        
        # Handle config
        self.config = config if config is not None else CustomDatasetConfig()
        
        # Store parameters
        self.split_set = split_set
        self.root_dir = root_dir
        self.num_points = num_points
        self.custom_scenes = all_scenes
        self.use_color = use_color
        self.use_height = use_height
        self.augment = augment if split_set == "train" else False
        # Get all available scenes
        print(len(self.custom_scenes))
        print("creating splits!!")
        # Create train/val split (80/20) if no predefined split exists
        split_idx = int(0.8 * len(self.custom_scenes))
        if split_set == "train":
            self.scene_files = self.custom_scenes[:split_idx]
        else:
            self.scene_files = self.custom_scenes[split_idx:]
        
        # Normalization parameters
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32)
        ]

    def __len__(self):
        return len(self.scene_files)

    # Angle normalization wrapper
    def normalize_angle(self,angle):
        """Constrain angle to [-π, π]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def normalize_pc(self,points_flat, raw_boxes): 
        centroid = np.mean(points_flat, axis=0)
        centered_points = points_flat - centroid

        scale = np.max(centered_points, axis=0) - np.min(centered_points, axis=0)
        normalized_points = centered_points / scale[np.newaxis, :]

        N = raw_boxes.shape[0]
        centroids_expanded = np.tile(
            np.repeat(centroid[np.newaxis, :], 8, axis=0)[np.newaxis, :, :],
            (N, 1, 1)
        )
        boxes_centered = raw_boxes - centroids_expanded
        scale_reshaped = scale.reshape(1, 1, 3)
        normalized_boxes = boxes_centered / scale_reshaped
        return normalized_points, normalized_boxes

    def extract_center_size_yaw_from_segments(self, pc_segments):
        results = []

        for seg in pc_segments:
            if seg.shape[0] < 4:
                results.append(None)
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(seg)
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

            obb = pcd.get_oriented_bounding_box()
            center = np.asarray(obb.center)
            size = np.asarray(obb.extent)
            R = np.asarray(obb.R)

            x_axis = R[:, 0]
            yaw = np.arctan2(-x_axis[1], x_axis[0])  # SUN RGB-D convention

            results.append(np.concatenate([center, size, [yaw]]))

        return np.array([r for r in results if r is not None])

    def __getitem__(self, idx):
        scene_path = self.scene_files[idx]
        subdir = os.path.dirname(scene_path)
        
        # Load data
        raw_pc = np.load(os.path.join(self.root_dir, subdir, "pc.npy")) #3,H,W
        pc = np.transpose(raw_pc, (1, 2, 0)).reshape(-1, 3) #H,W,3
        pc_sun = flip_axis_to_camera_np(pc) # original coordinates of scene (X=right, Y=down/forward, Z=down to SUN RGB-D coordinates (X=right, Y=forward, Z=up)

        raw_boxes = np.load(os.path.join(self.root_dir, subdir, "bbox3d.npy"))  # (N,8,3)
        rgb = np.array(Image.open(os.path.join(self.root_dir, subdir, "rgb.jpg"))) #(H, W, 3)
        masks = np.load(os.path.join(self.root_dir, subdir,"mask.npy"))
        rgb_pc = rgb.reshape(-1,3)/255.0
        boxes_sun = flip_axis_to_camera_np(raw_boxes.reshape(-1, 3)).reshape(-1, 8, 3)
        pc_sun_normalized, boxes_sun_normalized = self.normalize_pc(pc_sun,boxes_sun)
        pc_segments = []
        masks_flat = masks.reshape(masks.shape[0], -1)
        for i in range(masks.shape[0]):
          mask =masks_flat[i]
          instance_points = pc_sun_normalized[mask > 0]  # (num_pts, 3)
          pc_segments.append(instance_points)

        gt_boxes = self.extract_center_size_yaw_from_segments(pc_segments)

        # print("dumping pkl for vis.")
        # debug_path = os.path.join(self.root_dir, f'debug_sample_{idx}.pkl')
        # debug_data = {
        # 'scene_path': subdir,
        # 'pc': pc,
        # 'boxes' : raw_boxes,
        # 'pc_sun': pc_sun,
        # 'boxes_sun': boxes_sun,
        # 'rgb_pc' : rgb_pc,
        # 'gt_boxes' : gt_boxes
        # }

        # with open(debug_path, 'wb') as f: # used for Debugging
        #     pickle.dump(debug_data, f)

        # # breakpoint()


        # Initialize targets
        MAX_NUM_OBJ = self.config.max_num_obj
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6), dtype=np.float32)
        target_angles = np.zeros(MAX_NUM_OBJ, dtype=np.float32)
        angle_classes = np.zeros(MAX_NUM_OBJ, dtype=np.int64)
        angle_residuals = np.zeros(MAX_NUM_OBJ, dtype=np.float32)
        target_mask = np.zeros(MAX_NUM_OBJ, dtype=np.float32)

        valid_boxes = min(gt_boxes.shape[0], MAX_NUM_OBJ)
        target_bboxes[:valid_boxes] = gt_boxes[:valid_boxes, :6]
        target_mask[:valid_boxes] = 1

        # AT Train/Test time, do angle conversion with binning
        for i in range(valid_boxes):
            angle = gt_boxes[i, 6]
            angle_classes[i], angle_residuals[i] = self.config.angle2class(angle)
            target_angles[i] = self.config.class2angle(angle_classes[i], angle_residuals[i])

        # Augmentation
        if self.augment:
            # Flipping
            if np.random.rand() > 0.5:
                pc_sun_normalized[:, 0] = -pc_sun_normalized[:, 0]
                target_bboxes[:, 0] = -target_bboxes[:, 0]
                target_angles = np.array([self.config.class2angle(*self.config.angle2class(np.pi - a)) 
                                        for a in target_angles])

            if np.random.rand() > 0.5:
                pc_sun_normalized[:, 1] = -pc_sun_normalized[:, 1]
                target_bboxes[:, 1] = -target_bboxes[:, 1]
                target_angles = np.array([self.config.class2angle(*self.config.angle2class(-a)) 
                                        for a in target_angles])

            # Rotation
            rot_angle = (np.random.rand() * np.pi/9) - (np.pi/18)
            rot_mat = np.array([
                [np.cos(rot_angle), -np.sin(rot_angle), 0],
                [np.sin(rot_angle), np.cos(rot_angle), 0],
                [0, 0, 1]
            ])
            pc_sun_normalized[:, :3] = pc_sun_normalized[:, :3] @ rot_mat.T
            target_bboxes[:, :3] = target_bboxes[:, :3] @ rot_mat.T
            
            # Update angles using config's rotation handling
            target_angles = np.array([self.config.class2angle(*self.config.angle2class(a + rot_angle))
                                    for a in target_angles])
            angle_classes, angle_residuals = zip(*[self.config.angle2class(a) for a in target_angles])
            angle_classes, angle_residuals = np.array(angle_classes), np.array(angle_residuals)

        point_cloud, choices = pc_util.random_sampling(
            pc_sun_normalized, self.num_points, return_choices=True
        )

        # Normalization
        pc_min = point_cloud.min(axis=0)
        pc_max = point_cloud.max(axis=0)
        
        box_centers = target_bboxes[:, :3]
        box_centers_normalized = (box_centers - pc_min) / (pc_max - pc_min + 1e-6)
        box_sizes_normalized = target_bboxes[:, 3:6] / (pc_max - pc_min + 1e-6)

        # Generate box corners using config's method
        box_corners = self.config.box_parametrization_to_corners_np(
            box_centers[None, ...],
            target_bboxes[:, 3:6].astype(np.float32)[None, ...],
            target_angles.astype(np.float32)[None, ...]
        ).squeeze(0)

        return {
            "point_clouds": point_cloud.astype(np.float32),
            "gt_box_corners": box_corners.astype(np.float32),
            "gt_box_centers": box_centers.astype(np.float32),
            "gt_box_centers_normalized": box_centers_normalized.astype(np.float32),
            "gt_angle_class_label": angle_classes.astype(np.int64),
            "gt_angle_residual_label": angle_residuals.astype(np.float32),
            "gt_box_present": target_mask.astype(np.float32),
            "gt_box_sizes": target_bboxes[:, 3:6].astype(np.float32),
            "gt_box_sizes_normalized": box_sizes_normalized.astype(np.float32),
            "gt_box_angles": target_angles.astype(np.float32),
            "point_cloud_dims_min": pc_min.astype(np.float32),
            "point_cloud_dims_max": pc_max.astype(np.float32),
        }
