from typing import List
from enum import Enum
from ludwig.constants import TYPE, NUMBER, BINARY, CATEGORY, TEXT, IMAGE
from ludwig.utils.types import LudwigFeature


class ModelCategory(str, Enum):
    """The model category."""

    UNKNOWN = "unknown"
    TABULAR = "tabular"
    TEXT = "text"
    TABULAR_TEXT = "tabular_text"
    IMAGE = "image"
    TABULAR_IMAGE = "tabular_image"
    MULTI_MODAL = "multi_modal"


def contains_tabular_features(feature_types: List[str]):
    return NUMBER in feature_types or BINARY in feature_types or CATEGORY in feature_types


def get_model_category(input_features: List[LudwigFeature]) -> ModelCategory:
    input_feature_types = set()
    for input_feature in input_features:
        input_feature_types.add(input_feature[TYPE])

    if input_feature_types - {IMAGE, TEXT, NUMBER, BINARY, CATEGORY}:
        # There are unmapped feature types.
        return ModelCategory.UNKNOWN

    if IMAGE in input_feature_types and TEXT in input_feature_types and contains_tabular_features(input_feature_types):
        return ModelCategory.MULTI_MODAL
    if TEXT in input_feature_types and contains_tabular_features(input_feature_types):
        return ModelCategory.TABULAR_TEXT
    if IMAGE in input_feature_types and contains_tabular_features(input_feature_types):
        return ModelCategory.TABULAR_IMAGE
    if contains_tabular_features(input_feature_types):
        return ModelCategory.TABULAR
    if IMAGE in input_feature_types:
        return ModelCategory.IMAGE
    if TEXT in input_feature_types:
        return ModelCategory.TEXT

    return ModelCategory.UNKNOWN
