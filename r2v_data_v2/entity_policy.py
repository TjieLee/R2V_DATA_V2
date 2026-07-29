from __future__ import annotations

from r2v_data_v2.schemas import AnnotationEntity

_SEPARABLE_ENTITY_TYPES = {
    "composite_candidate",
    "independent",
    "important_independent_object",
}


def requires_foreground_mask(entity: AnnotationEntity) -> bool:
    return entity.separability in _SEPARABLE_ENTITY_TYPES
