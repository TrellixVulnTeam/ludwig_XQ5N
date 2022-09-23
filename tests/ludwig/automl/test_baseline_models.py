import yaml
from ludwig.automl.baseline_models import get_baseline_configs
from collections import defaultdict
from ludwig.automl.model_category import ModelCategory
from ludwig.api import LudwigModel
from ludwig import datasets

MULTI_MODAL_CONFIG = yaml.safe_load(
    """
    input_features:
      - name: default_profile
        type: binary
      - name: default_profile_image
        type: binary
      - name: description
        type: text
      - name: favourites_count
        type: number
      - name: followers_count
        type: number
      - name: friends_count
        type: number
      - name: geo_enabled
        type: binary
      - name: lang
        type: category
      - name: location
        type: category
      - name: profile_background_image_path
        type: category
      - name: profile_image_path
        type: image
        preprocessing:
          num_channels: 3
      - name: statuses_count
        type: number
      - name: verified
        type: binary
      - name: average_tweets_per_day
        type: number
      - name: account_age_days
        type: number
    output_features:
      - name: account_type
        type: binary
        """
)


def get_config_recommendations_model_category_count(config_recommendations):
    count = defaultdict(int)
    for config_recommendation in config_recommendations:
        count[config_recommendation.model_category] += 1
    return count


def get_num_recommendations_with_hyperopt(config_recommendations):
    count = 0
    for config_recommendation in config_recommendations:
        if config_recommendation.contains_hyperopt:
            count += 1
    return count


def test_get_baseline_configs_multi_modal():
    config_recommendations = get_baseline_configs(
        MULTI_MODAL_CONFIG["input_features"], MULTI_MODAL_CONFIG["output_features"]
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
