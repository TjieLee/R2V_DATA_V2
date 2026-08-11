"""Read-only H3-compatible sidecar preprocessing."""

from r2v_data_v2.h3.fusion import AudioBindingPolicy, build_audio_binding_sidecar
from r2v_data_v2.h3.schemas import AudioBindingSidecar, H3AudioBindingIR

__all__ = [
    "AudioBindingPolicy",
    "AudioBindingSidecar",
    "H3AudioBindingIR",
    "build_audio_binding_sidecar",
]
