import torch
import math


def my_bbox_iou(box1, box2, xywh=False, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):

    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

    # 算宽高
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    # 算交集和 IoU
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)

        # 对角线距离平方 & 中心点距离平方
        c2 = cw ** 2 + ch ** 2 + eps
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4

        if CIoU:
            focal_gamma = 0.5  # Focal 惩罚权重

            # 1. 计算 EIoU
            rho_w2 = (w2 - w1) ** 2
            rho_h2 = (h2 - h1) ** 2
            cw2 = cw ** 2 + eps
            ch2 = ch ** 2 + eps
            eiou = iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)


            eiou_loss = 1.0 - eiou
            focal_weight = iou.detach() ** focal_gamma
            focal_eiou_loss = focal_weight * eiou_loss

            return 1.0 - focal_eiou_loss

        return iou - rho2 / c2  # 兜底 DIoU

    return iou  # 兜底普通 IoU


# 2. 编写激活补丁的函数
def apply_focal_eiou_patch():
    import ultralytics.utils as metrics
    import ultralytics.utils.loss as loss_module

    metrics.bbox_iou = my_bbox_iou
    loss_module.bbox_iou = my_bbox_iou
