import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List

import yaml

from ludwig.automl import generally_good
from ludwig.automl.default_configs import (
    get_default_concat_model_with_features,
    get_default_gbm_model_with_features,
    get_default_tabnet_model_with_features,
)
from ludwig.automl.model_category import get_model_category, ModelCategory
from ludwig.automl.select_features import (
    get_image_features,
    get_tabular_features,
    get_tabular_text_features,
    get_text_features,
)
from ludwig.constants import BINARY, CATEGORY, INPUT_FEATURES, NUMBER, OUTPUT_FEATURES, TEXT
from ludwig.utils.types import LudwigConfig, LudwigFeature

logger = logging.getLogger(__name__)

# Set of output feature types that baseline configurations are supported for.
BASELINE_CONFIGS_SUPPORTED_OUTPUT_TYPES = {NUMBER, BINARY, CATEGORY}


@dataclass
class ConfigRecommendation:
    """Describes a ludwig configuration recommendation."""

    # Why this configuration is useful or interesting to try.
    description: str

    # The raw ludwig config, without hyperopt, backend, or execution-specific parameters.
    config: LudwigConfig

    # The model category this config represents, i.e. TABULAR, IMAGE, TEXT, TEXT_TABULAR, MULTI_MODAL, etc.
    model_category: ModelCategory

    # Whether the config includes a hyperopt section, and should run a hyperparameter search.
    contains_hyperopt: bool

    # Whether the config uses a pretrained model.
    uses_pretrained_model: bool


def _load_model_config(model_config_filename: str):
    """Loads a model config."""
    model_config_path = os.path.join(os.path.dirname(generally_good.__file__), model_config_filename)
    with open(model_config_path) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=3)
def load_generally_good_model_configs() -> Dict[str, Dict]:
    """Returns all model configs in the generally good directory, keyed by file name."""
    import importlib.resources

    config_filenames = [f for f in importlib.resources.contents(generally_good) if f.endswith(".yaml")]
    configs = {}
    for config_filename in config_filenames:
        basename = os.path.splitext(config_filename)[0]
        configs[basename] = _load_model_config(config_filename)
    return configs


def get_general_baseline_configs(
    input_features: List[Dict],
    output_features: List[Dict],
    model_category: ModelCategory,
) -> List[ConfigRecommendation]:
    """Returns a list of general baseline configs, without any feature pruning."""
    config_recommendations = []

    # Basic neural net using all features.
    config_recommendations.append(
        ConfigRecommendation(
            description="Basic neural network using all features.",
            model_category=model_category,
            config=get_default_concat_model_with_features(input_features, output_features),
            contains_hyperopt=False,
            uses_pretrained_model=False,
        ),
    )
    # General tabnet model using all features.
    config_recommendations.append(
        ConfigRecommendation(
            description="General tabnet model using all features.",
            model_category=model_category,
            config=get_default_tabnet_model_with_features(input_features, output_features),
            contains_hyperopt=False,
            uses_pretrained_model=False,
        ),
    )
    return config_recommendations


def get_text_tabular_input_baseline_model(
    tabular_text_input_features: List[LudwigFeature],
    output_features: List[LudwigFeature],
) -> LudwigConfig:
    """Returns a baseline models that use tabular and text features.

    Text features are encoded with BERT encoder.
    """
    # Use the generally good BERT training configuration.
    generally_good_model_configs = load_generally_good_model_configs()
    config = generally_good_model_configs["bert"]
    tabular_text_input_features_copy = deepcopy(tabular_text_input_features)
    for input_feature in tabular_text_input_features_copy:
        if input_feature["type"] == TEXT:
            input_feature["encoder"] = {}
            input_feature["encoder"]["type"] = "bert"

    config["input_features"] = tabular_text_input_features_copy
    config["output_features"] = output_features
    return config


def get_text_baseline_model(
    text_input_features: List[LudwigFeature],
    output_features: List[LudwigFeature],
    encoder_type: str,
) -> LudwigConfig:
    """Returns a config that encodes text features using the given encoder type."""
    generally_good_model_configs = load_generally_good_model_configs()
    config = {}
    if encoder_type == "bert":
        config = generally_good_model_configs["bert"]

    text_input_features_copy = deepcopy(text_input_features)
    for input_feature in text_input_features_copy:
        input_feature["encoder"] = {}
        input_feature["encoder"]["type"] = encoder_type

    config["input_features"] = text_input_features_copy
    config["output_features"] = output_features
    return config


def get_image_input_baseline_models(image_input_features, output_features) -> List[ConfigRecommendation]:
    """Returns a list of config recommendations for image models."""
    config_recommendations = []
    config = {}
    image_input_features_copy = deepcopy(image_input_features)
    for input_feature in image_input_features_copy:
        input_feature["encoder"] = {}
        input_feature["encoder"]["type"] = "vit"
    config["input_features"] = image_input_features_copy
    config["output_features"] = output_features
    config_recommendations.append(
        ConfigRecommendation(
            description="Pretrained VIT model",
            model_category=ModelCategory.IMAGE,
            config=config,
            contains_hyperopt=False,
            uses_pretrained_model=True,
        )
    )

    config_recommendations.append(
        ConfigRecommendation(
            description="Basic vision model using 1-layer CNN.",
            model_category=ModelCategory.IMAGE,
            config=get_default_concat_model_with_features(image_input_features, output_features),
            contains_hyperopt=False,
            uses_pretrained_model=False,
        )
    )
    return config_recommendations


def get_text_input_baseline_models(
    text_input_features: List[LudwigFeature], output_features: List[LudwigFeature]
) -> List[ConfigRecommendation]:
    """Returns a list of config recommendations for text baseline models."""
    config_recommendations = []
    config_recommendations.append(
        ConfigRecommendation(
            description="Basic text model with simple embeddings, whitespace tokenization.",
            model_category=ModelCategory.TEXT,
            config=get_text_baseline_model(text_input_features, output_features, encoder_type="embed"),
            contains_hyperopt=False,
            uses_pretrained_model=False,
        )
    )

    bert_config = get_text_baseline_model(text_input_features, output_features, encoder_type="bert")
    config_recommendations.append(
        ConfigRecommendation(
            description="Pretrained BERT model, without fine-tuning.",
            model_category=ModelCategory.TEXT,
            config=bert_config,
            contains_hyperopt=False,
            uses_pretrained_model=True,
        )
    )

    bert_config_with_fine_tuning = deepcopy(bert_config)
    for feature in bert_config_with_fine_tuning["input_features"]:
        feature["encoder"]["trainable"] = True
    config_recommendations.append(
        ConfigRecommendation(
            description="Pretrained BERT model, with fine-tuning.",
            model_category=ModelCategory.TEXT,
            config=bert_config_with_fine_tuning,
            contains_hyperopt=False,
            uses_pretrained_model=True,
        )
    )
    return config_recommendations


def get_generally_good_model_with_features(
    input_features: List[LudwigFeature], output_features: List[LudwigFeature], generally_good_model_key: str
):
    generally_good_model_configs = load_generally_good_model_configs()
    config = deepcopy(generally_good_model_configs[generally_good_model_key])
    config[INPUT_FEATURES] = input_features
    config[OUTPUT_FEATURES] = output_features
    return config


def get_tabular_model_baseline_configs_concise(
    tabular_input_features: List[LudwigFeature],
    output_features: List[LudwigFeature],
    generally_good_model_configs: Dict[str, LudwigConfig],
) -> List[ConfigRecommendation]:
    """Returns a list of config recommendations for a list of purely tabular input features."""
    config_recommendations = []
    # GBM models.
    config_recommendations.append(
        ConfigRecommendation(
            description=("Default GBM model, with hyperopt over num_boost_round and learning_rate."),
            model_category=ModelCategory.TABULAR,
            config=get_generally_good_model_with_features(tabular_input_features, output_features, "gbm_hyperopt"),
            contains_hyperopt=True,
            uses_pretrained_model=False,
        ),
    )
    # Basic neural net using tabular features.
    config_recommendations.append(
        ConfigRecommendation(
            description="Basic neural network using only tabular features.",
            model_category=ModelCategory.TABULAR,
            config=get_default_concat_model_with_features(tabular_input_features, output_features),
            contains_hyperopt=False,
            uses_pretrained_model=False,
        ),
    )
    # General tabnet model using tabular features.
    config_recommendations.append(
        ConfigRecommendation(
            description=(
                "Generally strong tabnet model with reasonable performance across a variety of tabular "
                "datasets, using only tabular features."
            ),
            model_category=ModelCategory.TABULAR,
            config=get_generally_good_model_with_features(tabular_input_features, output_features, "tabnet_1"),
            contains_hyperopt=False,
            uses_pretrained_model=False,
        ),
    )
    return config_recommendations


def get_tabular_model_baseline_configs(
    tabular_input_features: List[LudwigFeature],
    output_features: List[LudwigFeature],
    generally_good_model_configs: Dict[str, LudwigConfig],
) -> List[ConfigRecommendation]:
    """Same as the concise function, but with additional generally good models and default models."""
    config_recommendations = []
    # GBM models.
    config_recommendations.append(
        ConfigRecommendation(
            description=("Default GBM model, using only tabular features."),
            model_category=ModelCategory.TABULAR,
            config=get_default_gbm_model_with_features(tabular_input_features, output_features),
            contains_hyperopt=False,
            uses_pretrained_model=False,
        ),
    )
    config_recommendations.append(
        ConfigRecommendation(
            description=(
                "Generally strong GBM model with reasonable performance across a variety of tabular datasets, "
                "using only tabular features."
            ),
            model_category=ModelCategory.TABULAR,
            config=get_generally_good_model_with_features(tabular_input_features, output_features, "gbm_1"),
            contains_hyperopt=False,
            uses_pretrained_model=False,
        ),
    )
    config_recommendations.append(
        ConfigRecommendation(
            description=("Default GBM model, with hyperopt over num_boost_round and learning_rate."),
            model_category=ModelCategory.TABULAR,
            config=get_generally_good_model_with_features(tabular_input_features, output_features, "gbm_hyperopt"),
            contains_hyperopt=True,
            uses_pretrained_model=False,
        ),
    )

    # Basic neural net using tabular features.
    config_recommendations.append(
        ConfigRecommendation(
            description="Basic neural network using only tabular features.",
            model_category=ModelCategory.TABULAR,
            config=get_default_concat_model_with_features(tabular_input_features, output_features),
            contains_hyperopt=False,
            uses_pretrained_model=False,
        ),
    )
    config_recommendations.append(
        ConfigRecommendation(
            description=(
                "Generally strong basic neural network with reasonable performance across a variety of tabular "
                "datasets, using only tabular features."
            ),
            model_category=ModelCategory.TABULAR,
            config=get_generally_good_model_with_features(tabular_input_features, output_features, "concat_1"),
            contains_hyperopt=False,
            uses_pretrained_model=False,
        ),
    )
    # General tabnet model using tabular features.
    config_recommendations.append(
        ConfigRecommendation(
            description="General tabnet model using tabular features.",
            model_category=ModelCategory.TABULAR,
            config=get_default_tabnet_model_with_features(tabular_input_features, output_features),
            contains_hyperopt=False,
            uses_pretrained_model=False,
        ),
    )
    config_recommendations.append(
        ConfigRecommendation(
            description=(
                "Generally strong tabnet model with reasonable performance across a variety of tabular "
                "datasets, using only tabular features."
            ),
            model_category=ModelCategory.TABULAR,
            config=get_generally_good_model_with_features(tabular_input_features, output_features, "tabnet_1"),
            contains_hyperopt=False,
            uses_pretrained_model=False,
        ),
    )
    return config_recommendations


def get_multi_modal_baseline_configs(
    input_features: List[LudwigFeature],
    output_features: List[LudwigFeature],
    generally_good_model_configs: Dict[str, LudwigConfig],
):
    """Returns a list of config recommendations appropriate for multimodal models."""
    config_recommendations = []

    # Models that only use tabular features.
    tabular_input_features = get_tabular_features(input_features)
    if tabular_input_features:
        config_recommendations.extend(
            get_tabular_model_baseline_configs_concise(
                tabular_input_features, output_features, generally_good_model_configs
            )
        )

    # Models that only use text features.
    text_input_features = get_text_features(input_features)
    if text_input_features and len(input_features) > len(text_input_features):
        config_recommendations.extend(get_text_input_baseline_models(text_input_features, output_features))

    # Models that only use text and tabular features.
    tabular_text_input_features = get_tabular_text_features(input_features)
    if len(tabular_text_input_features) > len(tabular_input_features):
        tabular_text_model = get_text_tabular_input_baseline_model(tabular_text_input_features, output_features)
        config_recommendations.append(
            ConfigRecommendation(
                description="ECD defaults, with text and tabular features.",
                model_category=ModelCategory.TABULAR_TEXT,
                config=tabular_text_model,
                contains_hyperopt=False,
                uses_pretrained_model=False,
            )
        )

    # Models that only use image features.
    image_input_features = get_image_features(input_features)
    if image_input_features and len(input_features) > len(image_input_features):
        # Add a model that only uses the image input features.
        config_recommendations.extend(get_image_input_baseline_models(image_input_features, output_features))

    # Models that use all features.
    if len(input_features) > len(tabular_text_input_features):
        config_recommendations.extend(
            get_general_baseline_configs(input_features, output_features, ModelCategory.MULTI_MODAL)
        )
    return config_recommendations


def are_baseline_configs_supported(input_features, output_features):
    """Returns False if there are multiple output features, or if the type of the output feature isn't
    supported."""
    if len(output_features) > 1:
        return False
    if output_features[0]["type"] not in BASELINE_CONFIGS_SUPPORTED_OUTPUT_TYPES:
        return False
    return True


def get_baseline_configs(
    input_features: List[LudwigFeature], output_features: List[LudwigFeature]
) -> List[ConfigRecommendation]:
    """Returns a list of recommended config baselines with metadata describing them.

    As input, we take in a list of input and output features.

    We return a list of configurations that covers a broad variety of features, model types, and model architectures,
    paired with a human-readable description describing that configuration.

    We assume that the list of input features are all potentially useful, i.e. they have been sanitifix
    """
    if not are_baseline_configs_supported(input_features, output_features):
        return []

    # Load generally good models.
    generally_good_model_configs = load_generally_good_model_configs()

    model_category = get_model_category(input_features)
    if model_category == ModelCategory.MULTI_MODAL:
        return get_multi_modal_baseline_configs(input_features, output_features, generally_good_model_configs)

    if model_category == ModelCategory.TABULAR:
        return get_tabular_model_baseline_configs(input_features, output_features, generally_good_model_configs)

    # Fallback.
    return get_multi_modal_baseline_configs(input_features, output_features, generally_good_model_configs)
