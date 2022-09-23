import yaml
from ludwig.constants import INPUT_FEATURES
from ludwig.automl.model_category import ModelCategory, get_model_category
from tests.integration_tests.utils import SAMPLE_MULTI_MODAL_CONFIG


def test_get_model_category():
    assert get_model_category(SAMPLE_MULTI_MODAL_CONFIG[INPUT_FEATURES]) == ModelCategory.MULTI_MODAL
