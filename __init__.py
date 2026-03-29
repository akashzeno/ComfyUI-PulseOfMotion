"""ComfyUI-PulseOfMotion: Visual Chronometer PhyFPS prediction nodes.

Based on "Pulse of Motion" (https://github.com/taco-group/Pulse-of-Motion).
Predicts the Physical FPS (PhyFPS) of video frames — the true temporal scale
implied by visual motion dynamics.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
