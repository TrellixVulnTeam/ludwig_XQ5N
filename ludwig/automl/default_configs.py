import os
from typing import List, Dict
from functools import lru_cache
import yaml
from ludwig.automl import generally_good

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


def _load_model_config(model_config_filename: str):
    """Loads a model config."""
    model_config_path = os.path.join(os.path.dirname(generally_good.__file__), model_config_filename)
    with open(model_config_path) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=3)
def get_generally_good_model_configs() -> Dict[str, Dict]:
    """Returns all model configs for the specified dataset.
    Model configs are named <dataset_name>_<config_name>.yaml
    """
    import importlib.resources

    config_filenames = [f for f in importlib.resources.contents(generally_good) if f.endswith(".yaml")]
    configs = {}
    for config_filename in config_filenames:
        basename = os.path.splitext(config_filename)[0]
        configs[basename] = _load_model_config(config_filename)
    return configs
