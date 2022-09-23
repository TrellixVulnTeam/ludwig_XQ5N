"""Utilities for subselecting features."""

from typing import List, Set
from ludwig.utils.types import LudwigFeature
from ludwig.constants import NUMBER, CATEGORY, BINARY, TEXT, IMAGE


def get_features_with_type(features: List[LudwigFeature], accepted_types: Set[str]) -> List[LudwigFeature]:
    """Returns a list of features that are of one of the accepted types."""
    supported_features = []
    for feature in features:
        if feature["type"] in accepted_types:
            supported_features.append(feature)
    return supported_features


def get_tabular_features(features: List[LudwigFeature]) -> List[LudwigFeature]:
    """Returns list of tabular (NUMBER, CATEGORY, BINARY) features."""
    return get_features_with_type(features, {NUMBER, CATEGORY, BINARY})


def get_tabular_text_features(features: List[LudwigFeature]) -> List[LudwigFeature]:
    """Returns list of tabular and text features."""
    return get_features_with_type(features, {NUMBER, CATEGORY, BINARY, TEXT})


def get_text_features(features: List[LudwigFeature]) -> List[LudwigFeature]:
    """Returns list of text features."""
    return get_features_with_type(features, {TEXT})


def get_image_features(features: List[LudwigFeature]) -> List[LudwigFeature]:
    """Returns list of image features."""
    return get_features_with_type(features, {IMAGE})


def get_tabular_image_features(features: List[LudwigFeature]) -> List[LudwigFeature]:
    """Returns list of tabular and text features."""
    return get_features_with_type(features, {NUMBER, CATEGORY, BINARY, IMAGE})
