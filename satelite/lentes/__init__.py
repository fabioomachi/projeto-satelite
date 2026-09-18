"""Pacote Lentes — captura RTSP com leitura seletiva."""

from .camera_manager import CameraManager
from .motion import MotionSentinel

__all__ = ["CameraManager", "MotionSentinel"]
