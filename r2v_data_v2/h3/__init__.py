"""Read-only H3-compatible sidecar preprocessing."""

from r2v_data_v2.h3.audio_binding import (
    AudioBindingProductionConfig,
    build_audio_clip_binding_dataset,
    coalesce_audio_bindings,
)
from r2v_data_v2.h3.audio_pairing import (
    AudioPairingConfig,
    build_audio_pair_samples,
)
from r2v_data_v2.h3.audio_schemas import (
    AudioClipBinding,
    AudioPairSample,
    H3Sample,
)
from r2v_data_v2.h3.fusion import AudioBindingPolicy, build_audio_binding_sidecar
from r2v_data_v2.h3.primary_voice import (
    VoiceReferenceQualityPolicy,
    export_primary_voice_references,
)
from r2v_data_v2.h3.schemas import (
    AudioBindingSidecar,
    H3AudioBindingIR,
    H3TaskSpecification,
)

__all__ = [
    "AudioBindingPolicy",
    "AudioBindingProductionConfig",
    "AudioBindingSidecar",
    "AudioClipBinding",
    "AudioPairSample",
    "AudioPairingConfig",
    "H3AudioBindingIR",
    "H3Sample",
    "H3TaskSpecification",
    "VoiceReferenceQualityPolicy",
    "build_audio_binding_sidecar",
    "build_audio_clip_binding_dataset",
    "build_audio_pair_samples",
    "coalesce_audio_bindings",
    "export_primary_voice_references",
]
