"""
Models module for Human Detection Model.
Contains SSD architecture, backbone, and loss functions.
"""

from .ssd import SSD, create_ssd
from .backbone import MobileNetV2Backbone, ResNet50Backbone, create_backbone
from .ssd_head import SSDHead, get_num_priors_per_feature, calculate_total_priors
from .prior_box import PriorBox, generate_priors, decode_boxes, encode_boxes
from .loss import MultiBoxLoss

__all__ = [
    'SSD',
    'create_ssd',
    'MobileNetV2Backbone',
    'ResNet50Backbone',
    'create_backbone',
    'SSDHead',
    'get_num_priors_per_feature',
    'calculate_total_priors',
    'PriorBox',
    'generate_priors',
    'decode_boxes',
    'encode_boxes',
    'MultiBoxLoss'
]
