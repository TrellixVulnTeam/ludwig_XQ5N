from typing import List

from ludwig.constants import (
    MODEL_TYPE,
    MODEL_GBM,
    TRAINER,
    MODEL_ECD,
    COMBINER,
    TYPE,
    LIGHTGBM_TRAINER,
    INPUT_FEATURES,
    OUTPUT_FEATURES,
)
from ludwig.utils.types import LudwigConfig, LudwigFeature


def get_default_gbm_model() -> LudwigConfig:
    """Returns a default GBM model."""
    return {MODEL_TYPE: MODEL_GBM, TRAINER: {TYPE: LIGHTGBM_TRAINER}}


def get_default_gbm_model_with_features(
    input_features: List[LudwigFeature], output_features: List[LudwigFeature]
) -> LudwigConfig:
    """Returns a default GBM model with the specified features."""
    config = get_default_gbm_model()
    config[INPUT_FEATURES] = input_features
    config[OUTPUT_FEATURES] = output_features
    return config


def get_default_concat_model() -> LudwigConfig:
    """Returns a default concat model."""
    return {MODEL_TYPE: MODEL_ECD, COMBINER: {TYPE: "concat"}}


def get_default_concat_model_with_features(
    input_features: List[LudwigFeature], output_features: List[LudwigFeature]
) -> LudwigConfig:
    """Returns a default concat model with the specified features."""
    config = get_default_concat_model()
    config[INPUT_FEATURES] = input_features
    config[OUTPUT_FEATURES] = output_features
    return config


def get_default_tabnet_model() -> LudwigConfig:
    """Returns a default tabnet model."""
    return {MODEL_TYPE: MODEL_ECD, COMBINER: {TYPE: "tabnet"}, TRAINER: {TYPE: TRAINER}}


def get_default_tabnet_model_with_features(
    input_features: List[LudwigFeature], output_features: List[LudwigFeature]
) -> LudwigConfig:
    """Returns a default tabnet model with the specified features."""
    config = get_default_tabnet_model()
    config[INPUT_FEATURES] = input_features
    config[OUTPUT_FEATURES] = output_features
    return config
