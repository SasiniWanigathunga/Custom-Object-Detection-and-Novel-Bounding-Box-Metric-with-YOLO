import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import bbox_iou

def smooth_BCE(eps=0.1):  # label smoothing helper
    return 1.0 - 0.5 * eps, 0.5 * eps

class BCEBlurWithLogitsLoss(nn.Module):
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)
        dx = pred - true
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be BCEWithLogitsLoss
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class QFocalLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be BCEWithLogitsLoss
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# --- NEW: Custom Loss Class ---
class Custom_Loss(nn.Module):
    def __init__(self, beta=100):
        """
        Custom loss combining previous loss, center alignment loss, and aspect ratio loss,
        weighted by the predicted IoU.
        """
        super(Custom_Loss, self).__init__()
        self.beta = beta

    def forward(self, iou, prev_loss, center_loss, aspect_loss):
        term1 = prev_loss + center_loss + aspect_loss
        term2 = torch.exp(-self.beta * iou) * center_loss
        return term1 + term2

# --- Modified ComputeLoss Class ---
class ComputeLoss:
    def __init__(self, model, is_custom_loss=False, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device
        h = model.hyp

        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        self.cp, self.cn = smooth_BCE(eps=0.0)

        g = h['fl_gamma']
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])
        self.ssi = list(det.stride).index(16) if autobalance else 0
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))
        # Instantiate our custom loss module:
        self.custom_loss_fn = Custom_Loss(beta=10)
        self.is_custom_loss = is_custom_loss

    def __call__(self, p, targets):
        device = targets.device
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)

        # We'll accumulate additional losses for center and aspect ratio
        total_center_loss = torch.zeros(1, device=device)
        total_aspect_loss = torch.zeros(1, device=device)
        total_iou_sum = 0.0
        total_targets = 0

        eps = 1e-6

        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[..., 0], device=device)
            n = b.shape[0]
            if n:
                # Extract predictions corresponding to targets:
                ps = pi[b, a, gj, gi]  # shape: [n, 25]

                # Regression: predicted center and size
                pxy = ps[:, :2].sigmoid() * 2 - 0.5  # predicted center, [n, 2]
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]  # predicted width/height, [n, 2]
                pbox = torch.cat((pxy, pwh), 1)  # [n, 4]
                # Compute IoU between predicted and target boxes
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # shape: [n]
                lbox += (1.0 - iou).mean()

                # For custom loss, accumulate average IoU over targets
                total_iou_sum += iou.sum()
                total_targets += n

                # Compute center alignment loss: L1 loss between predicted center and target center
                center_loss_layer = torch.abs(pxy - tbox[i][:, :2]).mean()
                total_center_loss += center_loss_layer

                # Compute aspect ratio loss:
                pred_ratio = pwh[:, 0] / (pwh[:, 1] + eps)
                target_ratio = tbox[i][:, 2] / (tbox[i][:, 3] + eps)
                aspect_loss_layer = torch.abs(pred_ratio - target_ratio).mean()
                total_aspect_loss += aspect_loss_layer

                # Objectness target is set based on IoU:
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)

                # Classification loss (if multiple classes)
                if self.nc > 1:
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]
        original_loss = lbox + lobj + lcls

        # Average IoU over all targets:
        avg_iou = total_iou_sum / total_targets if total_targets > 0 else torch.tensor(0.0, device=device)

        # Compute final custom loss using our new formula:
        final_loss = self.custom_loss_fn(avg_iou, original_loss, total_center_loss, total_aspect_loss)

        if self.is_custom_loss == True:
            return final_loss * bs, torch.cat((original_loss, total_center_loss, final_loss)).detach()
        else:
            return original_loss * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

        g = 0.5
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float() * g

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]
            t = targets * gain
            if nt:
                r = t[:, :, 4:6] / anchors[:, None]
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']
                t = t[j]
                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            b, c = t[:, :2].long().T
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            gij = (gxy - offsets).long()
            gi, gj = gij.T
            a = t[:, 6].long()
            indices.append((b, a, gj.clamp_(0, int(gain[3]) - 1).long(), gi.clamp_(0, int(gain[2]) - 1).long()))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors[a])
            tcls.append(c)
        return tcls, tbox, indices, anch
