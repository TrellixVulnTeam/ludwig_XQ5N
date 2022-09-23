from ludwig.api import LudwigModel
from ludwig.automl.default_configs import get_default_concat_model_with_features
from ludwig.constants import INPUT_FEATURES, OUTPUT_FEATURES
from tests.integration_tests.utils import SAMPLE_MULTI_MODAL_CONFIG


def test_get_default_concat_model_with_features():
    config = get_default_concat_model_with_features(
        SAMPLE_MULTI_MODAL_CONFIG[INPUT_FEATURES], SAMPLE_MULTI_MODAL_CONFIG[OUTPUT_FEATURES]
    )

    # Check that the config is valid.
    LudwigModel(config)
