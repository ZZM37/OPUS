import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import MultiheadAttention, FFN
from mmdet.models import HEADS
from mmdet.models.utils.builder import TRANSFORMER

from ..bbox.utils import decode_points, encode_points
from ..utils import DUMP
from ..checkpoint import checkpoint as cp
from .opus_dcd_head import OPUSV2DCDHead


def _meta_to_tensor(values, reference):
    first_value = values[0]
    if isinstance(first_value, torch.Tensor):
        return torch.stack(
            [value.to(device=reference.device, dtype=reference.dtype) for value in values],
            dim=0,
        )
    return reference.new_tensor(np.asarray(values, dtype=np.float32))


def _first_available_meta(img_metas, keys, reference):
    for key in keys:
        if key in img_metas[0]:
            return _meta_to_tensor([meta[key] for meta in img_metas], reference)
    return None


def _to_4x4_matrix(matrix):
    if matrix is None:
        return None

    if matrix.shape[-2:] == (4, 4):
        return matrix

    if matrix.shape[-2:] == (3, 4):
        out_shape = (*matrix.shape[:-2], 4, 4)
        output = matrix.new_zeros(out_shape)
        output[..., :3, :4] = matrix
        output[..., 3, 3] = 1.0
        return output

    if matrix.shape[-2:] == (3, 3):
        out_shape = (*matrix.shape[:-2], 4, 4)
        output = matrix.new_zeros(out_shape)
        output[..., :3, :3] = matrix
        output[..., 3, 3] = 1.0
        return output

    raise ValueError(f"Unsupported matrix shape: {matrix.shape}")


def _build_fisheye_camera_cache(img_metas, reference, num_views):
    intrinsic = _first_available_meta(
        img_metas,
        [
            "fisheye_intrinsic",
            "fisheye_intrinsics",
            "camera_intrinsic",
            "camera_intrinsics",
            "cam_intrinsic",
            "cam_intrinsics",
            "cam2img",
            "cam2imgs",
        ],
        reference,
    )
    if intrinsic is None:
        raise KeyError(
            "Fisheye head requires one of "
            "`fisheye_intrinsic`, `camera_intrinsic`, `cam_intrinsic` or `cam2img` in img_metas."
        )
    intrinsic = intrinsic[..., :3, :3]

    distortion = _first_available_meta(
        img_metas,
        [
            "fisheye_distortion",
            "fisheye_distortions",
            "distortion",
            "distortions",
            "distortion_coeffs",
            "distortion_coefficients",
            "cam_distortion",
            "camera_distortion",
        ],
        reference,
    )
    if distortion is None:
        distortion = reference.new_zeros((len(img_metas), num_views, 4))
    distortion = distortion.reshape(len(img_metas), num_views, -1)
    if distortion.shape[-1] < 4:
        pad = distortion.new_zeros((*distortion.shape[:-1], 4 - distortion.shape[-1]))
        distortion = torch.cat([distortion, pad], dim=-1)
    distortion = distortion[..., :4]

    occ2cam = _first_available_meta(
        img_metas,
        ["occ2cam", "occ2cams", "occ2cam_rt", "occ2cam_rts"],
        reference,
    )
    if occ2cam is None:
        ego2cam = _first_available_meta(
            img_metas,
            ["ego2cam", "ego2cams", "ego2cam_rt", "ego2cam_rts"],
            reference,
        )
        if ego2cam is None:
            cam2ego = _first_available_meta(
                img_metas,
                ["cam2ego", "cam2egos", "cam2ego_rt", "cam2ego_rts"],
                reference,
            )
            if cam2ego is not None:
                ego2cam = torch.linalg.inv(_to_4x4_matrix(cam2ego))

        if ego2cam is None:
            ego2img = _first_available_meta(img_metas, ["ego2img"], reference)
            if ego2img is not None:
                k_pad = _to_4x4_matrix(intrinsic)
                ego2cam = torch.linalg.inv(k_pad) @ _to_4x4_matrix(ego2img)

        if ego2cam is None:
            raise KeyError(
                "Fisheye head requires `occ2cam`, `ego2cam`, `cam2ego`, or (`ego2img` + intrinsics) in img_metas."
            )

        ego2occ = _first_available_meta(img_metas, ["ego2occ"], reference)
        if ego2occ is None:
            raise KeyError("Fisheye head requires `ego2occ` in img_metas to convert occupancy points to ego frame.")
        occ2ego = torch.linalg.inv(_to_4x4_matrix(ego2occ))
        occ2cam = _to_4x4_matrix(ego2cam) @ occ2ego[:, None]
    else:
        occ2cam = _to_4x4_matrix(occ2cam)

    img_shape = _first_available_meta(img_metas, ["img_shape", "pad_shape", "ori_shape"], reference)
    if img_shape is None:
        raise KeyError("Fisheye head requires `img_shape`, `pad_shape`, or `ori_shape` in img_metas.")

    img_h = img_shape[..., 0]
    img_w = img_shape[..., 1]

    return dict(
        intrinsic=intrinsic[:, :num_views].contiguous(),
        distortion=distortion[:, :num_views].contiguous(),
        occ2cam=occ2cam[:, :num_views].contiguous(),
        img_h=img_h[:, :num_views].contiguous(),
        img_w=img_w[:, :num_views].contiguous(),
    )


def _masked_softmax(logits, mask, dim=-1, eps=1e-6):
    mask = mask.to(dtype=logits.dtype)
    logits = logits.masked_fill(mask == 0, -1e8)
    probs = torch.softmax(logits, dim=dim)
    probs = probs * mask
    return probs / probs.sum(dim=dim, keepdim=True).clamp_min(eps)


@TRANSFORMER.register_module()
class OPUSV2FisheyeTransformer(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_views=4,
                 num_points=4,
                 num_layers=6,
                 num_groups=4,
                 num_refines=[1, 2, 4, 8, 16, 32],
                 num_pt_channels=32,
                 scales=[1.0],
                 pc_range=[],
                 init_cfg=None):
        assert init_cfg is None, "init_cfg is not supported for OPUSV2FisheyeTransformer"
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.num_views = num_views
        self.num_layers = num_layers
        self.num_refines = num_refines
        self.num_pt_channels = num_pt_channels

        self.decoder = OPUSV2FisheyeDecoder(
            embed_dims=embed_dims,
            num_views=num_views,
            num_points=num_points,
            num_layers=num_layers,
            num_groups=num_groups,
            num_refines=num_refines,
            num_pt_channels=num_pt_channels,
            scales=scales,
            pc_range=pc_range,
        )

    @torch.no_grad()
    def init_weights(self):
        self.decoder.init_weights()

    def forward(self, query_points, query_feat, mlvl_feats, img_metas):
        pt_feats, refine_pts = self.decoder(query_points, query_feat, mlvl_feats, img_metas)
        pt_feats = [None if feat is None else torch.nan_to_num(feat) for feat in pt_feats]
        refine_pts = [None if pts is None else torch.nan_to_num(pts) for pts in refine_pts]
        return pt_feats, refine_pts


class OPUSV2FisheyeDecoder(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_views=4,
                 num_points=4,
                 num_layers=6,
                 num_groups=4,
                 num_refines=16,
                 num_pt_channels=32,
                 scales=[1.0],
                 pc_range=[],
                 init_cfg=None):
        super().__init__(init_cfg)
        self.num_layers = num_layers

        if len(scales) == 1:
            scales = scales * num_layers
        if not isinstance(num_refines, list):
            num_refines = [num_refines]
        if len(num_refines) == 1:
            num_refines = num_refines * num_layers
        before_refines = [1] + num_refines

        self.decoder_layers = ModuleList()
        for i in range(num_layers):
            self.decoder_layers.append(
                OPUSV2FisheyeDecoderLayer(
                    embed_dims=embed_dims,
                    num_views=num_views,
                    num_points=num_points,
                    num_groups=num_groups,
                    num_pt_channels=num_pt_channels,
                    num_refines=num_refines[i],
                    last_refines=before_refines[i],
                    last_layer=i == num_layers - 1,
                    scale=scales[i],
                    pc_range=pc_range,
                )
            )

    @torch.no_grad()
    def init_weights(self):
        self.decoder_layers.init_weights()

    def forward(self, query_points, query_feat, img_feat, img_metas):
        pt_feats, refine_pts = [], []
        camera_cache = _build_fisheye_camera_cache(
            img_metas=img_metas,
            reference=query_feat,
            num_views=img_feat.shape[1],
        )

        for i, decoder_layer in enumerate(self.decoder_layers):
            DUMP.stage_count = i
            query_points = query_points.detach()
            query_feat, pt_feat, query_points = decoder_layer(
                query_points, query_feat, img_feat, camera_cache
            )
            pt_feats.append(pt_feat)
            refine_pts.append(query_points)

        return pt_feats, refine_pts


class OPUSV2FisheyeDecoderLayer(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_views=4,
                 num_points=4,
                 num_groups=4,
                 num_pt_channels=32,
                 num_refines=16,
                 last_refines=16,
                 num_cls_fcs=2,
                 num_reg_fcs=2,
                 last_layer=False,
                 scale=1.0,
                 pc_range=[],
                 init_cfg=None):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self.num_refines = num_refines
        self.last_refines = last_refines
        self.last_layer = last_layer
        self.scale = scale

        self.position_encoder = nn.Sequential(
            nn.Linear(3 * self.last_refines, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.self_attn = OPUSFisheyeSelfAttention(embed_dims=embed_dims, num_heads=8, dropout=0.1, pc_range=pc_range)
        self.sampling = OPUSFisheyeSampling(
            embed_dims=embed_dims,
            num_views=num_views,
            num_groups=num_groups,
            num_points=num_points,
            pc_range=pc_range,
        )
        self.mixing = AdaptiveMixing(
            in_dim=embed_dims,
            in_points=num_points,
            n_groups=num_groups,
            out_points=32,
        )
        self.ffn = FFN(embed_dims, feedforward_channels=512, ffn_drop=0.1)

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

        cls_branch = []
        for _ in range(num_cls_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, num_pt_channels * self.num_refines))
        self.cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU(inplace=True))
        reg_branch.append(nn.Linear(self.embed_dims, 3 * self.num_refines))
        self.reg_branch = nn.Sequential(*reg_branch)

    @torch.no_grad()
    def init_weights(self):
        self.self_attn.init_weights()
        self.sampling.init_weights()
        self.mixing.init_weights()

    def refine_points(self, points_proposal, points_delta):
        bsz, num_query = points_delta.shape[:2]
        points_delta = points_delta.reshape(bsz, num_query, self.num_refines, 3)

        points_proposal = decode_points(points_proposal, self.pc_range)
        points_proposal = points_proposal.mean(dim=2, keepdim=True)
        new_points = points_proposal + points_delta
        return encode_points(new_points, self.pc_range)

    def forward(self, query_points, query_feat, img_feat, camera_cache):
        query_pos = self.position_encoder(query_points.flatten(2, 3))
        query_feat = query_feat + query_pos

        sampled_feat = self.sampling(query_points, query_feat, img_feat, camera_cache)
        query_feat = self.norm1(self.mixing(sampled_feat, query_feat))
        query_feat = self.norm2(self.self_attn(query_points, query_feat))
        query_feat = self.norm3(self.ffn(query_feat))

        bsz, num_query = query_points.shape[:2]
        reg_offset = self.scale * self.reg_branch(query_feat)
        refine_pt = self.refine_points(query_points, reg_offset)

        pt_feat = None
        if self.training or self.last_layer:
            pt_feat = self.cls_branch(query_feat)
            pt_feat = pt_feat.reshape(bsz, num_query, self.num_refines, -1)

        return query_feat, pt_feat, refine_pt


class OPUSFisheyeSelfAttention(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 pc_range=[],
                 init_cfg=None):
        super().__init__(init_cfg)
        self.pc_range = pc_range
        self.attention = MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        self.gen_tau = nn.Linear(embed_dims, num_heads)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def inner_forward(self, query_points, query_feat):
        dist = self.calc_points_dists(query_points)
        tau = self.gen_tau(query_feat)
        if DUMP.enabled:
            torch.save(tau.cpu(), "{}/sasa_tau_stage{}.pth".format(DUMP.out_dir, DUMP.stage_count))

        tau = tau.permute(0, 2, 1)
        attn_mask = dist[:, None, :, :] * tau[..., None]
        attn_mask = attn_mask.flatten(0, 1)
        return self.attention(query_feat, attn_mask=attn_mask)

    def forward(self, query_points, query_feat):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_points, query_feat, use_reentrant=False)
        return self.inner_forward(query_points, query_feat)

    @torch.no_grad()
    def calc_points_dists(self, points):
        points = decode_points(points, self.pc_range)
        points = points.mean(dim=2)
        dist = torch.norm(points.unsqueeze(-2) - points.unsqueeze(-3), dim=-1)
        return -dist


class OPUSFisheyeSampling(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 num_views=4,
                 num_groups=4,
                 num_points=8,
                 pc_range=[],
                 init_cfg=None):
        super().__init__(init_cfg)
        self.num_views = num_views
        self.num_groups = num_groups
        self.num_points = num_points
        self.pc_range = pc_range

        self.sampling_prototype = nn.Embedding(num_groups * num_points, 3)
        self.sampling_offset = nn.Linear(embed_dims, num_groups * num_points * 3)
        self.view_weights = nn.Linear(embed_dims, num_groups * num_points * num_views)

    def init_weights(self):
        bias = self.sampling_offset.bias.data.view(self.num_groups * self.num_points, 3)
        nn.init.normal_(self.sampling_prototype.weight, mean=0, std=1)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(bias[:, :3], -0.5, 0.5)
        nn.init.zeros_(self.view_weights.weight)
        nn.init.zeros_(self.view_weights.bias)

    def fisheye_project(self, sample_points, camera_cache, eps=1e-5):
        bsz, num_query, num_groups, num_points, _ = sample_points.shape
        num_views = camera_cache["occ2cam"].shape[1]
        num_samples = num_query * num_groups * num_points

        occ2cam = camera_cache["occ2cam"][:, :, None]
        intrinsic = camera_cache["intrinsic"][:, :, None]
        distortion = camera_cache["distortion"][:, :, None]
        img_h = camera_cache["img_h"][:, :, None]
        img_w = camera_cache["img_w"][:, :, None]

        sample_points = sample_points.reshape(bsz, 1, num_samples, 3)
        sample_points = torch.cat([sample_points, torch.ones_like(sample_points[..., :1])], dim=-1)
        sample_points = sample_points[..., None]

        cam_points = torch.matmul(occ2cam, sample_points).squeeze(-1)[..., :3]
        x_coord = cam_points[..., 0]
        y_coord = cam_points[..., 1]
        z_coord = cam_points[..., 2]

        valid = z_coord > eps
        norm_x = x_coord / z_coord.clamp_min(eps)
        norm_y = y_coord / z_coord.clamp_min(eps)
        radius = torch.sqrt(norm_x ** 2 + norm_y ** 2 + eps)

        theta = torch.atan(radius)
        theta2 = theta ** 2
        theta4 = theta2 ** 2
        theta6 = theta4 * theta2
        theta8 = theta4 ** 2

        k1, k2, k3, k4 = distortion.unbind(dim=-1)
        theta_d = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
        scale = torch.where(radius > eps, theta_d / radius, torch.ones_like(radius))

        x_distorted = norm_x * scale
        y_distorted = norm_y * scale

        fx = intrinsic[..., 0, 0]
        fy = intrinsic[..., 1, 1]
        skew = intrinsic[..., 0, 1]
        cx = intrinsic[..., 0, 2]
        cy = intrinsic[..., 1, 2]

        pixel_x = fx * x_distorted + skew * y_distorted + cx
        pixel_y = fy * y_distorted + cy

        valid = valid & (pixel_x >= 0) & (pixel_x <= (img_w - 1)) & (pixel_y >= 0) & (pixel_y <= (img_h - 1))

        grid_x = pixel_x / img_w.clamp_min(1.0) * 2.0 - 1.0
        grid_y = pixel_y / img_h.clamp_min(1.0) * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = grid.reshape(bsz, num_views, num_query, num_groups, num_points, 2)
        valid = valid.reshape(bsz, num_views, num_query, num_groups, num_points)
        return grid, valid

    def inner_forward(self, query_points, query_feat, img_feat, camera_cache):
        bsz, num_query = query_points.shape[:2]
        num_views = img_feat.shape[1]
        if num_views != self.num_views:
            raise ValueError(f"Expected {self.num_views} fisheye cameras, but got {num_views}.")

        if img_feat.shape[2] % self.num_groups != 0:
            raise ValueError("Image feature channels must be divisible by num_groups for grouped fisheye sampling.")

        query_points = decode_points(query_points, self.pc_range)
        if query_points.shape[2] == 1:
            query_center = query_points
            query_scale = torch.ones_like(query_center)
        else:
            query_center = query_points.mean(dim=2, keepdim=True)
            query_scale = query_points.std(dim=2, keepdim=True)

        sampling_offset = self.sampling_offset(query_feat).view(bsz, num_query, -1, 3)
        prototype = self.sampling_prototype.weight[None, None].expand(bsz, num_query, -1, -1)
        sampling_points = query_center + prototype * query_scale + sampling_offset
        sampling_points = sampling_points.view(bsz, num_query, self.num_groups, self.num_points, 3)

        grid, valid_mask = self.fisheye_project(sampling_points, camera_cache)

        _, _, channels, feat_h, feat_w = img_feat.shape
        group_channels = channels // self.num_groups
        img_feat = img_feat.view(bsz, num_views, self.num_groups, group_channels, feat_h, feat_w)
        img_feat = img_feat.permute(0, 2, 1, 3, 4, 5).reshape(bsz * self.num_groups * num_views, group_channels, feat_h, feat_w)

        grid = grid.permute(0, 3, 1, 2, 4, 5).reshape(bsz * self.num_groups * num_views, num_query, self.num_points, 2)
        sampled = F.grid_sample(
            img_feat,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.view(bsz, self.num_groups, num_views, group_channels, num_query, self.num_points)
        sampled = sampled.permute(0, 4, 1, 5, 2, 3)

        valid_mask = valid_mask.permute(0, 2, 3, 4, 1)
        view_logits = self.view_weights(query_feat).view(
            bsz, num_query, self.num_groups, self.num_points, self.num_views
        )
        view_weights = _masked_softmax(view_logits, valid_mask, dim=-1)
        sampled = (sampled * view_weights[..., None]).sum(dim=-2)
        return sampled

    def forward(self, query_points, query_feat, img_feat, camera_cache):
        if self.training and query_feat.requires_grad:
            return cp(
                self.inner_forward,
                query_points,
                query_feat,
                img_feat,
                camera_cache,
                use_reentrant=False,
            )
        return self.inner_forward(query_points, query_feat, img_feat, camera_cache)


class AdaptiveMixing(nn.Module):
    def __init__(self, in_dim, in_points, n_groups=1, query_dim=None, out_dim=None, out_points=None):
        super().__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points
        self.n_groups = n_groups
        self.out_dim = out_dim
        self.out_points = out_points

        self.eff_in_dim = in_dim // n_groups
        self.eff_out_dim = out_dim // n_groups
        self.m_parameters = self.eff_in_dim * self.eff_out_dim
        self.s_parameters = self.in_points * self.out_points
        self.total_parameters = self.m_parameters + self.s_parameters

        self.parameter_generator = nn.Linear(self.query_dim, self.n_groups * self.total_parameters)
        self.out_proj = nn.Linear(self.eff_out_dim * self.out_points * self.n_groups, self.query_dim)
        self.act = nn.ReLU(inplace=True)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.parameter_generator.weight)

    def inner_forward(self, x, query):
        bsz, num_query, num_groups, num_points, channels = x.shape
        assert num_groups == self.n_groups
        assert num_points == self.in_points
        assert channels == self.eff_in_dim

        params = self.parameter_generator(query).reshape(bsz * num_query, num_groups, -1)
        out = x.reshape(bsz * num_query, num_groups, num_points, channels)

        mix_channel, mix_point = params.split([self.m_parameters, self.s_parameters], dim=2)
        mix_channel = mix_channel.reshape(bsz * num_query, num_groups, self.eff_in_dim, self.eff_out_dim)
        mix_point = mix_point.reshape(bsz * num_query, num_groups, self.out_points, self.in_points)

        out = torch.matmul(out, mix_channel)
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        out = torch.matmul(mix_point, out)
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        out = out.reshape(bsz, num_query, -1)
        out = self.out_proj(out)
        return query + out

    def forward(self, x, query):
        if self.training and x.requires_grad:
            return cp(self.inner_forward, x, query, use_reentrant=False)
        return self.inner_forward(x, query)


@HEADS.register_module()
class OPUSV2FisheyeDCDHead(OPUSV2DCDHead):
    def forward(self, mlvl_feats, img_metas):
        if isinstance(mlvl_feats, (list, tuple)):
            if len(mlvl_feats) == 0:
                raise ValueError("`mlvl_feats` must contain at least one feature map.")
            # Only the first scale is used in the fisheye variant.
            img_feat = mlvl_feats[0]
        else:
            img_feat = mlvl_feats

        if img_feat.dim() != 5:
            raise ValueError(
                "Fisheye DCD head expects single-scale image features in shape [B, 4, C, H, W]."
            )

        batch_size = img_feat.shape[0]
        num_query = self.num_query
        init_points = self.init_points.weight[None, :, None, :].repeat(batch_size, num_query // self.num_query, 1, 1)
        query_feat = init_points.new_zeros(batch_size, num_query, self.embed_dims)

        pt_feats, refine_pts = self.transformer(
            init_points=init_points,
            query_feat=query_feat,
            mlvl_feats=img_feat,
            img_metas=img_metas,
        )

        cls_scores, voxel_coors = [], []
        for i, densifier in enumerate(self.densifiers):
            cls_score, voxel_coor = densifier(pt_feats[i], refine_pts[i])
            cls_scores.append(cls_score)
            voxel_coors.append(voxel_coor)

        return dict(
            init_points=init_points,
            all_refine_pts=refine_pts,
            all_cls_scores=cls_scores,
            all_voxel_coors=voxel_coors,
        )
