# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Helper functions and class to calculate Average Precisions for 3D object detection.
"""
import logging
import os
import sys
from collections import OrderedDict

import numpy as np
import scipy.special as scipy_special
import torch

from utils.box_util import (extract_pc_in_box3d, flip_axis_to_camera_np,
                            get_3d_box, get_3d_box_batch)
from utils.eval_det import eval_det_multiprocessing, get_iou_obb
from utils.nms import nms_2d_faster, nms_3d_faster, nms_3d_faster_samecls


def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # depth X,Y,Z = cam X,Z,-Y
    pc2[..., 2] *= -1
    return pc2


def softmax(x):
    """Numpy function for softmax"""
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs


# This is exactly the same as VoteNet so that we can compare evaluations.
def parse_predictions(
    predicted_boxes, point_cloud, config_dict
):
    pred_corners_3d_upright_camera = predicted_boxes.detach().cpu().numpy()

    K = pred_corners_3d_upright_camera.shape[1]  # K==num_proposal
    bsize = pred_corners_3d_upright_camera.shape[0]
    nonempty_box_mask = np.ones((bsize, K))

    if config_dict["remove_empty_box"]:
        batch_pc = point_cloud.cpu().numpy()[:, :, 0:3]  # B,N,3
        for i in range(bsize):
            pc = batch_pc[i, :, :]  # (N,3)
            for j in range(K):
                box3d = pred_corners_3d_upright_camera[i, j, :, :]  # (8,3)
                box3d = flip_axis_to_depth(box3d)
                pc_in_box, inds = extract_pc_in_box3d(pc, box3d)
                if len(pc_in_box) < 5:
                    nonempty_box_mask[i, j] = 0
            if nonempty_box_mask[i].sum() == 0:
                nonempty_box_mask[i, 0] = 1

    pred_mask = nonempty_box_mask

    batch_pred_map_cls = []
    for i in range(bsize):
        batch_pred_map_cls.append(
            [
                (0, pred_corners_3d_upright_camera[i, j], 1.0)
                for j in range(pred_corners_3d_upright_camera.shape[1])
                if pred_mask[i, j] == 1
            ]
        )

    return batch_pred_map_cls


def get_ap_config_dict(
    remove_empty_box=True,
    use_3d_nms=True,
    nms_iou=0.25,
    use_old_type_nms=False,
    cls_nms=True,
    per_class_proposal=True,
    use_cls_confidence_only=False,
    conf_thresh=0.05,
    no_nms=False,
    dataset_config=None,
):
    config_dict = {
        "remove_empty_box": remove_empty_box,
        "use_3d_nms": use_3d_nms,
        "nms_iou": nms_iou,
        "use_old_type_nms": use_old_type_nms,
        "cls_nms": cls_nms,
        "per_class_proposal": per_class_proposal,
        "use_cls_confidence_only": use_cls_confidence_only,
        "conf_thresh": conf_thresh,
        "no_nms": no_nms,
        "dataset_config": dataset_config,
    }
    return config_dict


class APCalculator(object):
    def __init__(
        self,
        dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=None,
        exact_eval=True,
        ap_config_dict=None,
    ):
        self.ap_iou_thresh = ap_iou_thresh
        if ap_config_dict is None:
            ap_config_dict = get_ap_config_dict(
                dataset_config=dataset_config, remove_empty_box=exact_eval
            )
        self.ap_config_dict = ap_config_dict
        self.class2type_map = class2type_map
        self.reset()
    def make_gt_list(self, gt_box_corners, gt_box_present):
        batch_gt_map_cls = []
        gt_box_corners = gt_box_corners.reshape(-1,64,8,3)
        bsize = gt_box_corners.shape[0]
        for i in range(bsize):
            batch_gt_map_cls.append(
                [
                    (0, gt_box_corners[i, j])
                    for j in range(gt_box_corners.shape[1])
                    if gt_box_present[i, j] == 1
                ]
            )
        return batch_gt_map_cls

    def step_meter(self, outputs, targets):
        if "outputs" in outputs:
            outputs = outputs["outputs"]
        self.step(
            predicted_box_corners=outputs["box_corners"],
            point_cloud=targets["point_clouds"],
            gt_box_corners=targets["gt_box_corners"],
            gt_box_present=targets["gt_box_present"],
        )

    def step(
        self,
        predicted_box_corners,
        point_cloud,
        gt_box_corners,
        gt_box_present,
    ):
        gt_box_corners = gt_box_corners.cpu().detach().numpy()
        gt_box_present = gt_box_present.cpu().detach().numpy()
        batch_gt_map_cls = self.make_gt_list(
            gt_box_corners, gt_box_present
        )

        batch_pred_map_cls = parse_predictions(
            predicted_box_corners,
            point_cloud,
            self.ap_config_dict,
        )

        self.accumulate(batch_pred_map_cls, batch_gt_map_cls)

    def accumulate(self, batch_pred_map_cls, batch_gt_map_cls):
        bsize = len(batch_pred_map_cls)
        assert bsize == len(batch_gt_map_cls)
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i]
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i]
            self.scan_cnt += 1

    def compute_metrics(self):
        overall_ret = OrderedDict()
        for ap_iou_thresh in self.ap_iou_thresh:
            ret_dict = OrderedDict()
            rec, prec, ap = eval_det_multiprocessing(
                self.pred_map_cls, self.gt_map_cls, ovthresh=ap_iou_thresh
            )
            for key in sorted(ap.keys()):
                clsname = self.class2type_map[key] if self.class2type_map else str(key)
                ret_dict["%s Average Precision" % (clsname)] = ap[key]
            ap_vals = np.array(list(ap.values()), dtype=np.float32)
            ap_vals[np.isnan(ap_vals)] = 0
            ret_dict["mAP"] = ap_vals.mean()
            rec_list = []
            for key in sorted(ap.keys()):
                clsname = self.class2type_map[key] if self.class2type_map else str(key)
                try:
                    ret_dict["%s Recall" % (clsname)] = rec[key][-1]
                    rec_list.append(rec[key][-1])
                except:
                    ret_dict["%s Recall" % (clsname)] = 0
                    rec_list.append(0)
            ret_dict["AR"] = np.mean(rec_list)
            overall_ret[ap_iou_thresh] = ret_dict
        return overall_ret

    def __str__(self):
        overall_ret = self.compute_metrics()
        return self.metrics_to_str(overall_ret)

    def metrics_to_str(self, overall_ret, per_class=True):
        mAP_strs = []
        AR_strs = []
        per_class_metrics = []
        for ap_iou_thresh in self.ap_iou_thresh:
            mAP = overall_ret[ap_iou_thresh]["mAP"] * 100
            mAP_strs.append(f"{mAP:.2f}")
            ar = overall_ret[ap_iou_thresh]["AR"] * 100
            AR_strs.append(f"{ar:.2f}")

            if per_class:
                per_class_metrics.append("-" * 5)
                per_class_metrics.append(f"IOU Thresh={ap_iou_thresh}")
                for x in list(overall_ret[ap_iou_thresh].keys()):
                    if x == "mAP" or x == "AR":
                        pass
                    else:
                        met_str = f"{x}: {overall_ret[ap_iou_thresh][x]*100:.2f}"
                        per_class_metrics.append(met_str)

        ap_header = [f"mAP{x:.2f}" for x in self.ap_iou_thresh]
        ap_str = ", ".join(ap_header)
        ap_str += ": " + ", ".join(mAP_strs)
        ap_str += "\n"

        ar_header = [f"AR{x:.2f}" for x in self.ap_iou_thresh]
        ap_str += ", ".join(ar_header)
        ap_str += ": " + ", ".join(AR_strs)

        if per_class:
            per_class_metrics = "\n".join(per_class_metrics)
            ap_str += "\n"
            ap_str += per_class_metrics

        return ap_str

    def metrics_to_dict(self, overall_ret):
        metrics_dict = {}
        for ap_iou_thresh in self.ap_iou_thresh:
            metrics_dict[f"mAP_{ap_iou_thresh}"] = (
                overall_ret[ap_iou_thresh]["mAP"] * 100
            )
            metrics_dict[f"AR_{ap_iou_thresh}"] = overall_ret[ap_iou_thresh]["AR"] * 100
        return metrics_dict

    def reset(self):
        self.gt_map_cls = {}
        self.pred_map_cls = {}
        self.scan_cnt = 0
