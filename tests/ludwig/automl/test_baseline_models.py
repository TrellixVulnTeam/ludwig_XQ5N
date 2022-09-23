from ludwig.automl.baseline_models import get_baseline_configs
from collections import defaultdict
from ludwig.automl.model_category import ModelCategory
from ludwig.api import LudwigModel
from ludwig import datasets
from tests.integration_tests.utils import SAMPLE_MULTI_MODAL_CONFIG


def get_config_recommendations_model_category_count(config_recommendations):
    """Returns a count of the model categories that were recommended."""
    count = defaultdict(int)
    for config_recommendation in config_recommendations:
        count[config_recommendation.model_category] += 1
    return count


def get_num_recommendations_with_hyperopt(config_recommendations):
    """Returns the number of model recommendations with hyperopt."""
    count = 0
    for config_recommendation in config_recommendations:
        if config_recommendation.contains_hyperopt:
            count += 1
    return count


def get_num_recommendations_with_pretrained_models(config_recommendations):
    """Returns the number of model recommendations that uses pretrained models."""
    count = 0
    for config_recommendation in config_recommendations:
        if config_recommendation.uses_pretrained_model:
            count += 1
    return count


def test_get_baseline_configs_multi_modal():
    config_recommendations = get_baseline_configs(
        SAMPLE_MULTI_MODAL_CONFIG["input_features"], SAMPLE_MULTI_MODAL_CONFIG["output_features"]
    )

    model_category_count = get_config_recommendations_model_category_count(config_recommendations)

    assert model_category_count == {
        ModelCategory.IMAGE: 2,
        ModelCategory.MULTI_MODAL: 2,
        ModelCategory.TABULAR: 3,
        ModelCategory.TABULAR_TEXT: 1,
        ModelCategory.TEXT: 3,
    }
    assert get_num_recommendations_with_hyperopt(config_recommendations) == 1
    assert get_num_recommendations_with_pretrained_models(config_recommendations) == 3

    # Check that all configs can be validly used to initialize a LudwigModel.
    for config_recommendation in config_recommendations:
        LudwigModel(config_recommendation.config)


def test_get_baseline_configs_all_datasets():
    for dataset_name in datasets.list_datasets():
        dataset = datasets.get_dataset(dataset_name)
        config = dataset.default_model_config

        if not config:
            continue

        config_recommendations = get_baseline_configs(config["input_features"], config["output_features"])

        # Check that all configs can be validly used to initialize a LudwigModel.
        for config_recommendation in config_recommendations:
            LudwigModel(config_recommendation.config)
