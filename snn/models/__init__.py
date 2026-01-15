"""SNN Models for Optical Flow"""

from .spiking_flownet import EventSNNFlowNetLite, EventSNNFlowNetLiteV2

__all__ = [
    'EventSNNFlowNetLite',
    'EventSNNFlowNetLiteV2',
]
