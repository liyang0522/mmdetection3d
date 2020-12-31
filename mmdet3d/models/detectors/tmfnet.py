import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F
from torch import nn as nn
import copy
import mmcv

from mmcv.parallel import DataContainer as DC

from os import path as osp



from mmdet3d.core import (Box3DMode, bbox3d2result, merge_aug_bboxes_3d,
                          show_result)
from mmdet3d.ops import Voxelization
from mmdet.core import multi_apply

from .. import builder

from mmdet.models import DETECTORS, build_backbone, build_head, build_neck, build_single_stage_head

from mmdet.models.detectors import BaseDetector



@DETECTORS.register_module()
class TaskMultiFusion(BaseDetector):
    """TaskMultiFusion Second using SimpleHardVFE ."""

    def __init__(self, 
    			img_backbone=None,
    			img_neck=None,
    			img_roi_head=None,
                img_rpn_head=None,
                pts_voxel_layer=None,
                pts_voxel_encoder=None,
                pts_middle_encoder=None,
                pts_fusion_layer=None,
                pts_backbone=None,
                pts_neck=None,
                pts_bbox_head=None,
                train_cfg=None,
                test_cfg=None,
                pretrained=None):
        super(TaskMultiFusion, self).__init__()

        if pts_voxel_layer:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        if pts_voxel_encoder:
            self.pts_voxel_encoder = builder.build_voxel_encoder(
                pts_voxel_encoder)
        if pts_middle_encoder:
            self.pts_middle_encoder = builder.build_middle_encoder(
                pts_middle_encoder)
        if pts_backbone:
            self.pts_backbone = builder.build_backbone(pts_backbone)
        if pts_fusion_layer:
            self.pts_fusion_layer = builder.build_fusion_layer(
                pts_fusion_layer)
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)
        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)

        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
        if img_rpn_head is not None:
            self.img_rpn_head = builder.build_head(img_rpn_head)
        if img_roi_head is not None:
            self.img_roi_head = builder.build_head(img_roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        super(TaskMultiFusion, self).init_weights(pretrained)
        if pretrained is None:
            img_pretrained = None
            pts_pretrained = None
        elif isinstance(pretrained, dict):
            img_pretrained = pretrained.get('img', None)
            pts_pretrained = pretrained.get('pts', None)
        else:
            raise ValueError(
                f'pretrained should be a dict, got {type(pretrained)}')
        if self.with_img_backbone:
            self.img_backbone.init_weights(pretrained=img_pretrained)
        if self.with_pts_backbone:
            self.pts_backbone.init_weights(pretrained=pts_pretrained)
        if self.with_img_neck:
            if isinstance(self.img_neck, nn.Sequential):
                for m in self.img_neck:
                    m.init_weights()
            else:
                self.img_neck.init_weights()

        if self.with_img_roi_head:
            self.img_roi_head.init_weights(img_pretrained)
        if self.with_img_rpn:
            self.img_rpn_head.init_weights()
        if self.with_pts_bbox:
            self.pts_bbox_head.init_weights()

    @property
    def with_img_shared_head(self):
        """bool: Whether the detector has a shared head in image branch."""
        return hasattr(self,
                       'img_shared_head') and self.img_shared_head is not None

    @property
    def with_pts_bbox(self):
        """bool: Whether the detector has a 3D box head."""
        return hasattr(self,
                       'pts_bbox_head') and self.pts_bbox_head is not None

    @property
    def with_img_bbox(self):
        """bool: Whether the detector has a 2D image box head."""
        return hasattr(self,
                       'img_bbox_head') and self.img_bbox_head is not None

    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_pts_backbone(self):
        """bool: Whether the detector has a 3D backbone."""
        return hasattr(self, 'pts_backbone') and self.pts_backbone is not None

    @property
    def with_fusion(self):
        """bool: Whether the detector has a fusion layer."""
        return hasattr(self,
                       'pts_fusion_layer') and self.fusion_layer is not None

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_pts_neck(self):
        """bool: Whether the detector has a neck in 3D detector branch."""
        return hasattr(self, 'pts_neck') and self.pts_neck is not None

    @property
    def with_img_rpn(self):
        """bool: Whether the detector has a 2D RPN in image detector branch."""
        return hasattr(self, 'img_rpn_head') and self.img_rpn_head is not None

    @property
    def with_img_roi_head(self):
        """bool: Whether the detector has a RoI Head in image branch."""
        return hasattr(self, 'img_roi_head') and self.img_roi_head is not None

    @property
    def with_voxel_encoder(self):
        """bool: Whether the detector has a voxel encoder."""
        return hasattr(self,
                       'voxel_encoder') and self.voxel_encoder is not None

    @property
    def with_middle_encoder(self):
        """bool: Whether the detector has a middle encoder."""
        return hasattr(self,
                       'middle_encoder') and self.middle_encoder is not None

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def extract_feat(self, points, img_metas=None):
        """Directly extract features from the backbone+neck.

        Args:
            points (torch.Tensor): Input points.
        """
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.pts_neck:
            x = self.pts_neck(x)
        return x
        
    def extract_feats(self, points, img_metas):
        """Extract features of multiple samples."""
        return [
            self.extract_feat(pts, img_meta)
            for pts, img_meta in zip(points, img_metas)
        ]
	
    def forward_train(self,
    	img_metas, 
    	imgs, 
    	points, 
    	gt_bboxes_3d,
    	gt_labels_3d,
    	gt_bboxes_ignore=None,
    	**kwargs):

   		x = self.extract_feat(points, img_metas)
   		outs = self.pts_bbox_head(x)
   		loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
   		losses = self.pts_bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
   		return losses
    

    def forward_test(self, img, img_meta, **kwargs):

        batch_size = len(img_meta)

        ret = self.merge_second_batch(kwargs)

        vx = self.backbone(ret['voxels'], ret['num_points'])
        (x, conv6) = self.neck(vx, ret['coordinates'], batch_size, is_test=True)

        rpn_outs = self.rpn_head.forward(x)

        guided_anchors, anchor_labels = self.rpn_head.get_guided_anchors(*rpn_outs, ret['anchors'], ret['anchors_mask'],
                                                                       None, None, thr=.1)

        bbox_score = self.extra_head(conv6, guided_anchors, is_test=True)

        det_bboxes, det_scores, det_labels = self.extra_head.get_rescore_bboxes(
            guided_anchors, bbox_score, anchor_labels, img_meta, self.test_cfg.extra)

        results = [kitti_bbox2results(*param, class_names=self.class_names) for param in zip(det_bboxes, det_scores, det_labels, img_meta)]

        return results
    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function without augmentaiton."""
        x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        feats = self.extract_feats(points, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]