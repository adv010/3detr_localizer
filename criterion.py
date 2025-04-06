import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.box_util import generalized_box3d_iou
from utils.dist import all_reduce_average
from utils.misc import huber_loss
from scipy.optimize import linear_sum_assignment
import numpy as np

class Matcher(nn.Module):
    def __init__(self, cost_objectness, cost_giou, cost_center):
        super().__init__()
        self.cost_objectness = cost_objectness
        self.cost_giou = cost_giou
        self.cost_center = cost_center

    @torch.no_grad()
    def forward(self, outputs, targets):
        batchsize = outputs["center_dist"].shape[0]
        nqueries = outputs["center_dist"].shape[1]
        ngt = targets["gt_box_centers_normalized"].shape[1]
        nactual_gt = targets["nactual_gt"]

        # objectness_mat = -outputs["center_dist"].unsqueeze(-1)
        center_mat = outputs["center_dist"].detach()
        giou_mat = -outputs["gious"].detach()

        final_cost = (
            # self.cost_objectness * objectness_mat
            + self.cost_center * center_mat
            + self.cost_giou * giou_mat
        )

        final_cost = final_cost.detach().cpu().numpy()
        assignments = []

        batch_size, nprop = final_cost.shape[0], final_cost.shape[1]
        per_prop_gt_inds = torch.zeros(
            [batch_size, nprop], dtype=torch.int64, device=outputs["center_dist"].device
        )
        proposal_matched_mask = torch.zeros(
            [batch_size, nprop], dtype=torch.float32, device=outputs["center_dist"].device
        )
        for b in range(batchsize):
            assign = []
            if nactual_gt[b] > 0:
                assign = linear_sum_assignment(final_cost[b, :, : nactual_gt[b]])
                assign = [
                    torch.from_numpy(x).long().to(device=outputs["center_dist"].device)
                    for x in assign
                ]
                per_prop_gt_inds[b, assign[0]] = assign[1]
                proposal_matched_mask[b, assign[0]] = 1
            assignments.append(assign)

        return {
            "assignments": assignments,
            "per_prop_gt_inds": per_prop_gt_inds,
            "proposal_matched_mask": proposal_matched_mask,
        }


class SetCriterion(nn.Module):
    def __init__(self, matcher, dataset_config, loss_weight_dict):
        super().__init__()
        self.dataset_config = dataset_config
        self.matcher = matcher
        self.loss_weight_dict = loss_weight_dict

        self.loss_functions = {
            "loss_angle": self.loss_angle,
            "loss_center": self.loss_center,
            "loss_size": self.loss_size,
            "loss_giou": self.loss_giou
            # "loss_cardinality": self.loss_cardinality,
        }

    # @torch.no_grad()
    # def loss_cardinality(self, outputs, targets, assignments):
    #     pred_objects = (outputs["objectness_prob"] > 0.5).sum(1)
    #     card_err = F.l1_loss(pred_objects.float(), targets["nactual_gt"])
    #     return {"loss_cardinality": card_err}


    def loss_angle(self, outputs, targets, assignments):
        angle_logits = outputs["angle_logits"]
        angle_residual = outputs["angle_residual_normalized"]

        if targets["num_boxes_replica"] > 0:
            gt_angle_label = targets["gt_angle_class_label"]
            gt_angle_residual = targets["gt_angle_residual_label"]
            gt_angle_residual_normalized = gt_angle_residual / (
                np.pi / self.dataset_config.num_angle_bin
            )
            gt_angle_label = torch.gather(
                gt_angle_label, 1, assignments["per_prop_gt_inds"]
            )
            angle_cls_loss = F.cross_entropy(
                angle_logits.transpose(2, 1), gt_angle_label, reduction="none"
            )
            angle_cls_loss = (
                angle_cls_loss * assignments["proposal_matched_mask"]
            ).sum()

            gt_angle_residual_normalized = torch.gather(
                gt_angle_residual_normalized, 1, assignments["per_prop_gt_inds"]
            )
            gt_angle_label_one_hot = torch.zeros_like(
                angle_residual, dtype=torch.float32
            )
            gt_angle_label_one_hot.scatter_(2, gt_angle_label.unsqueeze(-1), 1)

            angle_residual_for_gt_class = torch.sum(
                angle_residual * gt_angle_label_one_hot, -1
            )
            angle_reg_loss = huber_loss(
                angle_residual_for_gt_class - gt_angle_residual_normalized, delta=1.0
            )
            angle_reg_loss = (
                angle_reg_loss * assignments["proposal_matched_mask"]
            ).sum()

            angle_cls_loss /= targets["num_boxes"]
            angle_reg_loss /= targets["num_boxes"]
        else:
            angle_cls_loss = torch.zeros(1, device=angle_logits.device).squeeze()
            angle_reg_loss = torch.zeros(1, device=angle_logits.device).squeeze()
        return {"loss_angle_cls": angle_cls_loss, "loss_angle_reg": angle_reg_loss}
    
    def loss_center(self, outputs, targets, assignments):
        center_dist = outputs["center_dist"]
        if targets["num_boxes_replica"] > 0:
            center_loss = torch.gather(
                center_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
            ).squeeze(-1)
            center_loss = center_loss * assignments["proposal_matched_mask"]
            center_loss = center_loss.sum() / targets["num_boxes"]
        else:
            center_loss = torch.zeros(1, device=center_dist.device).squeeze()
        return {"loss_center": center_loss}

    def loss_size(self, outputs, targets, assignments):
        gt_box_sizes = targets["gt_box_sizes_normalized"]
        pred_box_sizes = outputs["size_normalized"]

        if targets["num_boxes_replica"] > 0:
            gt_box_sizes = torch.stack([
                torch.gather(gt_box_sizes[:, :, x], 1, assignments["per_prop_gt_inds"])
                for x in range(gt_box_sizes.shape[-1])
            ], dim=-1)
            size_loss = F.l1_loss(pred_box_sizes, gt_box_sizes, reduction="none").sum(dim=-1)
            size_loss = size_loss * assignments["proposal_matched_mask"]
            size_loss = size_loss.sum() / targets["num_boxes"]
        else:
            size_loss = torch.zeros(1, device=pred_box_sizes.device).squeeze()
        return {"loss_size": size_loss}

    def loss_giou(self, outputs, targets, assignments):
        gious_dist = 1 - outputs["gious"]
        giou_loss = torch.gather(
            gious_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
        ).squeeze(-1)
        giou_loss = giou_loss * assignments["proposal_matched_mask"]
        giou_loss = giou_loss.sum() / targets["num_boxes"]
        return {"loss_giou": giou_loss}

    def single_output_forward(self, outputs, targets):
        gious = generalized_box3d_iou(
            outputs["box_corners"],
            targets["gt_box_corners"],
            targets["nactual_gt"],
            rotated_boxes=torch.any(targets["gt_box_angles"] > 0).item(),
            needs_grad=(self.loss_weight_dict["loss_giou_weight"] > 0),
        )
        outputs["gious"] = gious
        center_dist = torch.cdist(
            outputs["center_normalized"], targets["gt_box_centers_normalized"], p=1
        )
        outputs["center_dist"] = center_dist
        assignments = self.matcher(outputs, targets)

        losses = {}
        for k in self.loss_functions:
            loss_wt_key = k + "_weight"
            if (
                loss_wt_key in self.loss_weight_dict
                and self.loss_weight_dict[loss_wt_key] > 0
            ) or loss_wt_key not in self.loss_weight_dict:
                curr_loss = self.loss_functions[k](outputs, targets, assignments)
                losses.update(curr_loss)

        final_loss = 0
        for k in self.loss_weight_dict:
            if self.loss_weight_dict[k] > 0:
                losses[k.replace("_weight", "")] *= self.loss_weight_dict[k]
                final_loss += losses[k.replace("_weight", "")]
        return final_loss, losses

    def forward(self, outputs, targets):
        nactual_gt = targets["gt_box_present"].sum(axis=1).long()
        num_boxes = torch.clamp(all_reduce_average(nactual_gt.sum()), min=1).item()
        targets["nactual_gt"] = nactual_gt
        targets["num_boxes"] = num_boxes
        targets["num_boxes_replica"] = nactual_gt.sum().item()

        loss, loss_dict = self.single_output_forward(outputs["outputs"], targets)

        if "aux_outputs" in outputs:
            for k in range(len(outputs["aux_outputs"])):
                interm_loss, interm_loss_dict = self.single_output_forward(
                    outputs["aux_outputs"][k], targets
                )
                loss += interm_loss
                for interm_key in interm_loss_dict:
                    loss_dict[f"{interm_key}_{k}"] = interm_loss_dict[interm_key]
        return loss, loss_dict


def build_criterion(args, dataset_config):
    matcher = Matcher(
        cost_giou=args.matcher_giou_cost,
        cost_center=args.matcher_center_cost,
        cost_objectness=args.matcher_objectness_cost,
    )

    loss_weight_dict = {
        "loss_giou_weight": args.loss_giou_weight,
        "loss_angle_cls_weight": args.loss_angle_cls_weight,
        "loss_angle_reg_weight": args.loss_angle_reg_weight,
        "loss_center_weight": args.loss_center_weight,
        "loss_size_weight": args.loss_size_weight,
    }
    criterion = SetCriterion(matcher, dataset_config, loss_weight_dict)
    return criterion
