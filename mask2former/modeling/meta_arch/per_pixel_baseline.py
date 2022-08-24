# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..transformer_decoder.maskformer_transformer_decoder import build_transformer_decoder

from ..transformer_decoder.maskformer_transformer_decoder import StandardTransformerDecoder
from ..pixel_decoder.fpn import build_pixel_decoder

def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


@SEM_SEG_HEADS_REGISTRY.register()
class PerPixelBaselineHead(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            logger = logging.getLogger(__name__)
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "sem_seg_head" in k and not k.startswith(prefix + "predictor"):
                    newk = k.replace(prefix, prefix + "pixel_decoder.")
                    # logger.warning(f"{k} ==> {newk}")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        num_points: int,
        oversample_ratio,
        importance_sample_ratio,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor = Conv2d(
            self.pixel_decoder.mask_dim, num_classes, kernel_size=1, stride=1, padding=0
        )
        weight_init.c2_msra_fill(self.predictor)

        # point rend params
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "num_points": cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            "oversample_ratio": cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            "importance_sample_ratio": cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        }

    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        x = self.layers(features)
        if self.training:
            return None, self.losses(x, targets)
        else:
            x = F.interpolate(
                x, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            return x, {}

    def layers(self, features):
        x, _, _ = self.pixel_decoder.forward_features(features)
        x = self.predictor(x)
        return x

    def losses(self, predictions, targets):
        predictions = predictions.float()  # https://github.com/pytorch/pytorch/issues/48163
        
        predictions = predictions[:, None]
        targets = targets[:, None]
        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                predictions,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                targets,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            predictions,
            point_coords,
            align_corners=False,
        ).squeeze(1)
        
        loss = F.cross_entropy(
            point_logits, point_labels, reduction="mean", ignore_index=self.ignore_value
        )
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses


@SEM_SEG_HEADS_REGISTRY.register()
class PerPixelBaselinePlusHead(PerPixelBaselineHead):
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "sem_seg_head" in k and not k.startswith(prefix + "predictor"):
                    newk = k.replace(prefix, prefix + "pixel_decoder.")
                    logger.debug(f"{k} ==> {newk}")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        # extra parameters
        transformer_predictor: nn.Module,
        transformer_in_feature: str,
        deep_supervision: bool,
        # inherit parameters
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
            deep_supervision: whether or not to add supervision to the output of
                every transformer decoder layer
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
        """
        super().__init__(
            input_shape,
            num_classes=num_classes,
            pixel_decoder=pixel_decoder,
            loss_weight=loss_weight,
            ignore_value=ignore_value,
        )

        del self.predictor

        self.predictor = transformer_predictor

        # I don't know why, but transformer_predictor is being given as a tuple, TODO: investigate this
        if isinstance(self.predictor, tuple):
            self.predictor = self.predictor[0]
        
        self.transformer_in_feature = transformer_in_feature
        self.deep_supervision = deep_supervision

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = super().from_config(cfg, input_shape)
        ret["transformer_in_feature"] = cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE
        if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "transformer_encoder":
            in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "multi_scale_pixel_decoder":
            in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        else:
            in_channels = input_shape[ret["transformer_in_feature"]].channels
        ret["transformer_predictor"] = build_transformer_decoder(
                cfg,
                in_channels,
                mask_classification=False,
            ),
        ret["deep_supervision"] = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        return ret

    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        x, aux_outputs = self.layers(features)
        if not self.training:
            if self.deep_supervision:
                losses = self.losses(x, targets)
                for i, aux_output in enumerate(aux_outputs):
                    losses["loss_sem_seg" + f"_{i}"] = self.losses(
                        aux_output["pred_masks"], targets
                    )["loss_sem_seg"]
                return None, losses
            else:
                return None, self.losses(x, targets)
        else:
            x = F.interpolate(
                x, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            return x, {}

    def layers(self, features):


        mask_features, transformer_encoder_features, _ = self.pixel_decoder.forward_features(features)
        if self.transformer_in_feature == "transformer_encoder":
            assert (
                transformer_encoder_features is not None
            ), "Please use the TransformerEncoderPixelDecoder."
            predictions = self.predictor(transformer_encoder_features, mask_features)
        elif self.transformer_in_feature == "multi_scale_pixel_decoder":
            
            mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)
            
            predictions = self.predictor(multi_scale_features, mask_features, None)
        else:
            predictions = self.predictor(features[self.transformer_in_feature], mask_features)
        if self.deep_supervision:
            return predictions["pred_masks"], predictions["aux_outputs"]
        else:
            return predictions["pred_masks"], None
