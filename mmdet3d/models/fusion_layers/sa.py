import torch
from mmcv.cnn import ConvModule, xavier_init
from torch import nn as nn
from torch.nn import functional as F


from ..registry import FUSION_LAYERS
from ...ops.pointnet_modules import pointnet2_modules as pointnet2_stack_modules
from ...ops.pointnet_modules import pointnet2_utils as pointnet2_stack_utils

@FUSION_LAYERS.register_module()
class SA(nn.Module):
 
    def __init__(self,
                 mlps,
                 pool_radius,
                 nsample):
        super(SA, self).__init__()
      
        self.mlps = mlps
        self.pool_radius = pool_radius
        self.nsample = nsample
        
       
        for k in range(len(self.mlps)):
            self.mlps[k] = [1] + self.mlps[k]

        self.SA_rawpoints = pointnet2_stack_modules.StackSAModuleMSG(
            radii=self.pool_radius,
            nsamples=self.nsample,
            mlps=self.mlps,
            use_xyz=True,
            pool_method='max_pool',
            )
        
        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of modules."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                xavier_init(m, distribution='uniform')

    def forward(self, voxel_center, voxel_coords, pts):
        """Forward function.

        Args:
            img_feats (list[torch.Tensor]): Image features.
            pts: [list[torch.Tensor]]: A batch of points with shape N x 3.
            pts_feats (torch.Tensor): A tensor consist of point features of the
                total batch.
            img_metas (list[dict]): Meta information of images.

        Returns:
            torch.Tensor: Fused features of each point.
        """
       

         #*******************************
        #voxel_center sa from  points
        

        batch_size = voxel_coords[-1, 0] + 1
        #print('batch_size',batch_size) #1

        new_xyz = voxel_center  #(num_voxel, 3)
        #print('voxel_center shape',)
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            new_xyz_batch_cnt[bs_idx] = (voxel_coords[:, 0] == bs_idx).sum()
        #print(new_xyz_batch_cnt)
        pts = pts[0]
        #print('pts shape',pts.shape) #(19196,4)
        #print('pts type',pts[0])
        xyz = pts[:, 0:3]
        #print('xyz len',len(xyz))  #19196
        #print(xyz)
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = len(xyz)
        #print(xyz_batch_cnt)
        point_features = pts[:, 3:].contiguous() 
        #print('11111111')
        #print(point_features)
        pooled_points, pooled_features = self.SA_rawpoints(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features,
        )
        #print('2222')  
        
        return pooled_features

    