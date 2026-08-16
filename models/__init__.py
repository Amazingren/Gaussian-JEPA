"""Model registry exposed by the public Gaussian-JEPA release."""

from .build import MODELS, build_model_from_cfg
from .Gaussian_JEPA_ExpMultiScale import (
    Gaussian_JEPA_ExpMultiScale,
    PointTransformer_JEPA_ExpMultiScale,
)

# Keep legacy checkpoint class names intact while providing
# concise public names for new configuration files.
MODELS.register_module(name="GaussianJEPA", module=Gaussian_JEPA_ExpMultiScale)
MODELS.register_module(
    name="PointTransformer_GaussianJEPA",
    module=PointTransformer_JEPA_ExpMultiScale,
)
