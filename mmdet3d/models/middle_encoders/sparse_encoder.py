from mmcv.runner import auto_fp16
from torch import nn as nn
import torch
from mmcv.runner import force_fp32

import torch.nn.functional as F
from mmdet3d.ops import SparseBasicBlock, make_sparse_convmodule
from mmdet3d.ops import spconv as spconv
from ..registry import MIDDLE_ENCODERS
from mmdet3d.ops.pointnet3 import pointnet3_utils
from mmdet3d.ops import pts_in_boxes3d
from mmdet3d.models.losses import weighted_smoothl1, weighted_sigmoid_focal_loss
def tensor2points(i,tensor, offset=(0., -40., -3.), voxel_size=(.05, .05, .1)):
    indices = tensor.indices.float() #torch.float16
    offset = torch.Tensor(offset).to(indices.device)
    voxel_size = torch.Tensor(voxel_size).to(indices.device)
    indices[:, 1:] = indices[:, [3, 2, 1]] * voxel_size * (i+1) + offset + .5 * voxel_size * (i+1)
    #print('tensor.features dtype',tensor.features.dtype) #torch.float16
    #print('indices dtype',indices.dtype) #torch.float16
    return tensor.features, indices

def point_sample(
    img_features,
    points,
    lidar2img_rt,
    pcd_rotate_mat,
    img_scale_factor,
    img_crop_offset,
    pcd_trans_factor,
    pcd_scale_factor,
    pcd_flip,
    img_flip,
    img_pad_shape,
    img_shape,
    aligned=True,
    padding_mode='zeros',
    align_corners=True,
):
    """Obtain image features using points.
    Args:
        img_features (torch.Tensor): 1 x C x H x W image features.
        points (torch.Tensor): Nx3 point cloud in LiDAR coordinates.
        lidar2img_rt (torch.Tensor): 4x4 transformation matrix.
        pcd_rotate_mat (torch.Tensor): 3x3 rotation matrix of points
            during augmentation.
        img_scale_factor (torch.Tensor): Scale factor with shape of \
            (w_scale, h_scale).
        img_crop_offset (torch.Tensor): Crop offset used to crop \
            image during data augmentation with shape of (w_offset, h_offset).
        pcd_trans_factor ([type]): Translation of points in augmentation.
        pcd_scale_factor (float): Scale factor of points during.
            data augmentation
        pcd_flip (bool): Whether the points are flipped.
        img_flip (bool): Whether the image is flipped.
        img_pad_shape (tuple[int]): int tuple indicates the h & w after
            padding, this is necessary to obtain features in feature map.
        img_shape (tuple[int]): int tuple indicates the h & w before padding
            after scaling, this is necessary for flipping coordinates.
        aligned (bool, optional): Whether use bilinear interpolation when
            sampling image features for each point. Defaults to True.
        padding_mode (str, optional): Padding mode when padding values for
            features of out-of-image points. Defaults to 'zeros'.
        align_corners (bool, optional): Whether to align corners when
            sampling image features for each point. Defaults to True.
    Returns:
        torch.Tensor: NxC image features sampled by point coordinates.
    """
    # aug order: flip -> trans -> scale -> rot
    # The transformation follows the augmentation order in data pipeline
    if pcd_flip:
        # if the points are flipped, flip them back first
        points[:, 1] = -points[:, 1]
    points -= pcd_trans_factor
    # the points should be scaled to the original scale in velo coordinate
    points /= pcd_scale_factor
    # the points should be rotated back
    # pcd_rotate_mat @ pcd_rotate_mat.inverse() is not exactly an identity
    # matrix, use angle to create the inverse rot matrix neither.
    points = points @ pcd_rotate_mat.float().inverse().half()

    # project points from velo coordinate to camera coordinate
    num_points = points.shape[0]
    pts_4d = torch.cat([points, points.new_ones(size=(num_points, 1))], dim=-1)
    pts_2d = pts_4d @ lidar2img_rt.t()

    # cam_points is Tensor of Nx4 whose last column is 1
    # transform camera coordinate to image coordinate

    pts_2d[:, 2] = torch.clamp(pts_2d[:, 2], min=1e-5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]

    # img transformation: scale -> crop -> flip
    # the image is resized by img_scale_factor
    img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2
    img_coors -= img_crop_offset

    # grid sample, the valid grid range should be in [-1,1]
    coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1

    if img_flip:
        # by default we take it as horizontal flip
        # use img_shape before padding for flip
        orig_h, orig_w = img_shape
        coor_x = orig_w - coor_x

    h, w = img_pad_shape
    coor_y = coor_y / h * 2 - 1
    coor_x = coor_x / w * 2 - 1
    grid = torch.cat([coor_x, coor_y],
                     dim=1).unsqueeze(0).unsqueeze(0)  # Nx2 -> 1x1xNx2

    # align_corner=True provides higher performance
    mode = 'bilinear' if aligned else 'nearest'
    #print('img_features',img_features.dtype)
    #print('grid',grid.dtype)
    grid = grid.type_as(img_features)
    #print('grid',grid.dtype)
    point_features = F.grid_sample(
        img_features,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners)  # 1xCx1xN feats

    return point_features.squeeze().t()

def sample_single(img_feats, pts, img_meta):
        """Sample features from single level image feature map.
        Args:
            img_feats (torch.Tensor): Image feature map in shape
                (N, C, H, W).
            pts (torch.Tensor): Points of a single sample.
            img_meta (dict): Meta information of the single sample.
        Returns:
            torch.Tensor: Single level image features of each point.
        """
        pts = pts[:,1:4].half()
        pcd_scale_factor = (
            img_meta['pcd_scale_factor']
            if 'pcd_scale_factor' in img_meta.keys() else 1)
        pcd_trans_factor = (
            pts.new_tensor(img_meta['pcd_trans'])
            if 'pcd_trans' in img_meta.keys() else 0)
        pcd_rotate_mat = (
            pts.new_tensor(img_meta['pcd_rotation']) if 'pcd_rotation'
            in img_meta.keys() else torch.eye(3).type_as(pts).to(pts.device))
        img_scale_factor = (
            pts.new_tensor(img_meta['scale_factor'][:2])
            if 'scale_factor' in img_meta.keys() else 1)
        pcd_flip = img_meta['pcd_flip'] if 'pcd_flip' in img_meta.keys(
        ) else False
        img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
        img_crop_offset = (
            pts.new_tensor(img_meta['img_crop_offset'])
            if 'img_crop_offset' in img_meta.keys() else 0)
        img_pts = point_sample(
            img_feats,
            pts,
            pts.new_tensor(img_meta['lidar2img']),
            pcd_rotate_mat,
            img_scale_factor,
            img_crop_offset,
            pcd_trans_factor,
            pcd_scale_factor,
            pcd_flip=pcd_flip,
            img_flip=img_flip,
            img_pad_shape=img_meta['input_shape'][:2],
            img_shape=img_meta['img_shape'][:2],
            aligned=True,
            padding_mode='zeros',
            align_corners=False
        )
        return img_pts

class Fusion_Conv(nn.Module):
    def __init__(self, inplanes, outplanes):

        super(Fusion_Conv, self).__init__()

        self.conv1 = torch.nn.Conv1d(inplanes, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)

    def forward(self, point_features, img_features):
        #print(point_features.shape, img_features.shape)
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features

#================addition attention (add)=======================#
class IA_Layer(nn.Module):
    def __init__(self, channels):
        
        super(IA_Layer, self).__init__()
        self.ic, self.pc = channels
        rc = self.ic // 4

        self.conv1 = nn.Sequential(nn.Conv1d(self.ic, self.pc, 1),
                                    nn.BatchNorm1d(self.pc),
                                    nn.ReLU())
        self.fc1 = nn.Linear(self.ic, rc)
        self.fc2 = nn.Linear(self.pc, rc)
        self.fc3 = nn.Linear(rc, 1)


    def forward(self, img_feas, point_feas):
        #batch = img_feas.size(0)
        #img_feas_f = img_feas.transpose(1,2).contiguous().view(-1, self.ic) #BCN->BNC->(BN)C
        #point_feas_f = point_feas.transpose(1,2).contiguous().view(-1, self.pc) #BCN->BNC->(BN)C'
        # print(img_feas)
        ri = self.fc1(img_feas)
        rp = self.fc2(point_feas)
        #print('ri shape',ri.shape)
        #print('rp shape',rp.shape)
        #print('rp+ri shape',(rp+ri).shape)
        att = torch.sigmoid(self.fc3(torch.tanh(ri + rp))) #BNx1
        #print('att',att.shape) #att torch.Size([16000, 1])
        att = att.squeeze(1)
        att = att.view(1, 1, -1) #B1N
        # print(img_feas.size(), att.size())
        #print('att',att.shape)
        img_feas = img_feas.unsqueeze(0).transpose(1,2)
        img_feas_new = self.conv1(img_feas)

        #print('img_feas_new shape', img_feas_new.shape) #([1, 16, 16000])
        out = img_feas_new * att
        #print('out shape', out.shape)


        return out


class Atten_Fusion_Conv(nn.Module):
    def __init__(self, inplanes_I, inplanes_P, outplanes):
        super(Atten_Fusion_Conv, self).__init__()

        self.IA_Layer = IA_Layer(channels = [inplanes_I, inplanes_P])
        # self.conv1 = torch.nn.Conv1d(inplanes_P, outplanes, 1)
        self.conv1 = torch.nn.Conv1d(inplanes_P + inplanes_P, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)


    def forward(self, point_features, img_features):
        # print(point_features.shape, img_features.shape)
        
        img_features =  self.IA_Layer(img_features, point_features)
        #print("img_features:", img_features.shape)

        #fusion_features = img_features + point_features
        #print('point_features shape',point_features.shape)
        point_features = point_features.unsqueeze(0).transpose(1,2)
        fusion_features = torch.cat([point_features, img_features], dim=1)
        #print('fusion_features shape',fusion_features.shape) #[1, 32, 16000]) ([1, 64, 28791]) [1, 128, 20122]) [1, 128, 9285])


        fusion_features = F.relu(self.bn1(self.conv1(fusion_features))) #[1, 16, 16000]) [1, 32, 28791]) [1, 64, 20122]) [1, 64, 20122])

       #print('fusion_features 2222 shape',fusion_features.shape)
        fusion_features = fusion_features.squeeze(0).transpose(0,1)
        return fusion_features

def nearest_neighbor_interpolate(unknown, known, known_feats):
    """
    :param pts: (n, 4) tensor of the bxyz positions of the unknown features
    :param ctr: (m, 4) tensor of the bxyz positions of the known features
    :param ctr_feats: (m, C) tensor of features to be propigated
    :return:
        new_features: (n, C) tensor of the features of the unknown features
    """
    
    unknown = unknown.contiguous().float()
    known = known.contiguous().float()
    known_feats = known_feats.float()
    dist, idx = pointnet3_utils.three_nn(unknown, known)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm
    interpolated_feats = pointnet3_utils.three_interpolate(known_feats, idx, weight)
    #interpolated_feats = interpolated_feats.half()
    return interpolated_feats



@MIDDLE_ENCODERS.register_module()
class SparseEncoder(nn.Module):
    r"""Sparse encoder for SECOND and Part-A2.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (list[str]): Order of conv module. Defaults to ('conv',
            'norm', 'act').
        norm_cfg (dict): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]]):
            Convolutional channels of each encode block.
        encoder_paddings (tuple[tuple[int]]): Paddings of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        block_type (str): Type of the block to use. Defaults to 'conv_module'.
    """

    def __init__(self,
                 in_channels,
                 sparse_shape,
                 order=('conv', 'norm', 'act'),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 base_channels=16,
                 output_channels=128,
                 encoder_channels=((16, ), (32, 32, 32), (64, 64, 64), (64, 64,
                                                                        64)),
                 encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1,
                                                                 1)),
                 block_type='conv_module',
                 IMG_CHANNELS = [3, 256, 256, 256, 256],
                 POINT_CHANNELS = [16, 32, 64, 64],
                 ADD_Image_Attention=True):
        super().__init__()
        assert block_type in ['conv_module', 'basicblock']
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False
        self.add_image_attention = ADD_Image_Attention
        # Spconv init all weight on its own

        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {'conv', 'norm', 'act'}
        self.Fusion_Conv = nn.ModuleList()
        for i in range(len(IMG_CHANNELS) - 1):
            if self.add_image_attention:
                    self.Fusion_Conv.append(
                            Atten_Fusion_Conv(IMG_CHANNELS[i + 1], POINT_CHANNELS[i], POINT_CHANNELS[i]))
        
        self.point_fc = nn.Linear(176, 64, bias=False)
        self.point_cls = nn.Linear(64, 1, bias=False)
        self.point_reg = nn.Linear(64, 3, bias=False)

        if self.order[0] != 'conv':  # pre activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d',
                order=('conv', ))
        else:  # post activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d')

        encoder_out_channels = self.make_encoder_layers(
            make_sparse_convmodule,
            norm_cfg,
            self.base_channels,
            block_type=block_type)

        self.conv_out = make_sparse_convmodule(
            encoder_out_channels,
            self.output_channels,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            norm_cfg=norm_cfg,
            padding=0,
            indice_key='spconv_down2',
            conv_type='SparseConv3d')

    def build_aux_target(self, nxyz, gt_boxes3d, enlarge=1.0):
        #print('gt_boxes3d all',len(gt_boxes3d))
        center_offsets = list()
        pts_labels = list()
        for i in range(len(gt_boxes3d)):
            #print('gt_boxes3d',gt_boxes3d[i].tensor.shape)
            #rint('gt_boxes3d',gt_boxes3d[i].tensor)
            boxes3d = gt_boxes3d[i].tensor.cpu()
            idx = torch.nonzero(nxyz[:, 0] == i).view(-1)
            new_xyz = nxyz[idx, 1:].cpu()

            boxes3d[:, 3:6] *= enlarge
            # pts_in_flag = torch.IntTensor(M, N).fill_(0) N = len(pts) M = len(boxes3d)

            pts_in_flag, center_offset = pts_in_boxes3d(new_xyz.float(), boxes3d.float())
            pts_label = pts_in_flag.max(0)[0].byte()

            # import mayavi.mlab as mlab
            # from mmdet.datasets.kitti_utils import draw_lidar, draw_gt_boxes3d
            # f = draw_lidar((new_xyz).numpy(), show=False)
            # pts = new_xyz[pts_label].numpy()
            # mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], color=(1, 1, 1), scale_factor=0.25, figure=f)
            # f = draw_gt_boxes3d(center_to_corner_box3d(boxes3d.numpy()), f, draw_text=False, show=True)

            pts_labels.append(pts_label)
            center_offsets.append(center_offset)

        center_offsets = torch.cat(center_offsets).cuda()
        pts_labels = torch.cat(pts_labels).cuda()

        return pts_labels, center_offsets

    
    def aux_loss(self, points, point_cls, point_reg, gt_bboxes):

        N = len(gt_bboxes)

        pts_labels, center_targets = self.build_aux_target(points, gt_bboxes)

        rpn_cls_target = pts_labels.float()
        pos = (pts_labels > 0).float()
        neg = (pts_labels == 0).float()

        pos_normalizer = pos.sum()
        pos_normalizer = torch.clamp(pos_normalizer, min=1.0)

        cls_weights = pos + neg
        cls_weights = cls_weights / pos_normalizer

        reg_weights = 1 / point_cls.sigmoid() * pts_labels
        reg_weights = reg_weights / pos_normalizer

        aux_loss_cls = weighted_sigmoid_focal_loss(point_cls.view(-1), rpn_cls_target, weight=cls_weights, avg_factor=1.)
        aux_loss_cls /= N

        aux_loss_reg = weighted_smoothl1(point_reg, center_targets, beta=1 / 9., weight=reg_weights[..., None], avg_factor=1.)
        aux_loss_reg /= N

        return dict(
            aux_loss_cls = aux_loss_cls,
            aux_loss_reg = aux_loss_reg,
        )


    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size,img_feats, img_metas, is_test=False):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.float32): Voxel features in shape (N, C).
            coors (torch.int32): Coordinates in shape (N, 4), \
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            dict: Backbone features.
        """
        #print('img_feats',img_feats[0].shape) #[1, 256, 88, 288]
        #print('img_feats',img_feats[1].shape)   #[1, 512, 84, 272])
        #print('img_feats',img_feats[2].shape) #([1, 1024, 42, 136])
        #print('img_feats',img_feats[3].shape) #[1, 2048, 21, 68])

        points_mean = torch.zeros_like(voxel_features)
        points_mean[:, 0] = coors[:, 0]
        points_mean[:, 1:] = voxel_features[:, :3]

        coors = coors.int()
        input_sp_tensor = spconv.SparseConvTensor(voxel_features, coors,
                                                  self.sparse_shape,
                                                  batch_size)
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        middle = []
        i=0
        for encoder_layer in self.encoder_layers:

            x = encoder_layer(x)
           
            #print('x dtype',x.dtype)
            indices = x.indices

            spatial_shape = x.spatial_shape
            batch_size = x.batch_size
            

            vx_feat, vx_nxyz = tensor2points(i, x, (0, -40., -3.))
            print('vx_feat dtype',vx_feat.dtype)
            print('vx_nxyz dtype',vx_nxyz.dtype)
            img_feat = img_feats[i]
            print('img_feat dtype',img_feat.dtype)
            point_img = sample_single(img_feat, vx_nxyz, img_metas[0])
            print('point_img dtype',point_img.dtype)
            fusion_features = self.Fusion_Conv[i](vx_feat, point_img)
            print('fusion_features dtype',fusion_features.dtype)
            #x = spconv.SparseConvTensor(fusion_features, indices,spatial_shape, batch_size)
            #x.features = fusion_features
            print('x111 features dtype',x.features.dtype)
            print('x111 features shape',x.features.shape)
            print('indices1111 dtype',indices.dtype)
            i += 1
            middle.append((vx_nxyz, fusion_features))
            encode_features.append(x)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(encode_features[-1])
        spatial_features = out.dense()

        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)

        if is_test:
            return spatial_features
        else:
            vx_nxyz, vx_feat = middle[0]
            p0 = nearest_neighbor_interpolate(points_mean, vx_nxyz, vx_feat.contiguous())
            
            vx_nxyz, vx_feat = middle[1]
            p1 = nearest_neighbor_interpolate(points_mean, vx_nxyz, vx_feat.contiguous())
            
            vx_nxyz, vx_feat = middle[2]
            p2 = nearest_neighbor_interpolate(points_mean, vx_nxyz, vx_feat.contiguous())
            
            vx_nxyz, vx_feat = middle[3]
            p3 = nearest_neighbor_interpolate(points_mean, vx_nxyz, vx_feat.contiguous())
            
            pointwise = self.point_fc(torch.cat([p0, p1, p2, p3], dim=-1))
            point_cls = self.point_cls(pointwise)
            point_reg = self.point_reg(pointwise)

            return spatial_features, (points_mean, point_cls, point_reg)

    def make_encoder_layers(self,
                            make_block,
                            norm_cfg,
                            in_channels,
                            block_type='conv_module',
                            conv_cfg=dict(type='SubMConv3d')):
        """make encoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_cfg (dict[str]): Config of normalization layer.
            in_channels (int): The number of encoder input channels.
            block_type (str): Type of the block to use. Defaults to
                'conv_module'.
            conv_cfg (dict): Config of conv layer. Defaults to
                dict(type='SubMConv3d').

        Returns:
            int: The number of encoder output channels.
        """
        assert block_type in ['conv_module', 'basicblock']
        self.encoder_layers = spconv.SparseSequential()

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                # each stage started with a spconv layer
                # except the first stage
                if i != 0 and j == 0 and block_type == 'conv_module':
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            stride=2,
                            padding=padding,
                            indice_key=f'spconv{i + 1}',
                            conv_type='SparseConv3d'))
                elif block_type == 'basicblock':
                    if j == len(blocks) - 1 and i != len(
                            self.encoder_channels) - 1:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                stride=2,
                                padding=padding,
                                indice_key=f'spconv{i + 1}',
                                conv_type='SparseConv3d'))
                    else:
                        blocks_list.append(
                            SparseBasicBlock(
                                out_channels,
                                out_channels,
                                norm_cfg=norm_cfg,
                                conv_cfg=conv_cfg))
                else:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            padding=padding,
                            indice_key=f'subm{i + 1}',
                            conv_type='SubMConv3d'))
                in_channels = out_channels
            stage_name = f'encoder_layer{i + 1}'
            stage_layers = spconv.SparseSequential(*blocks_list)
            self.encoder_layers.add_module(stage_name, stage_layers)
        return out_channels
