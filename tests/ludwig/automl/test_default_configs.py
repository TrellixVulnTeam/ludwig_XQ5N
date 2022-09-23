import yaml

from ludwig.constants import INPUT_FEATURES, OUTPUT_FEATURES
from ludwig.automl.default_configs import get_default_gbm_model_with_features, get_generally_good_model_configs
from ludwig.api import LudwigModel


def test_get_default_gbm_model_with_features():
    config = yaml.safe_load(
        """
    input_features:
      - name: default_profile
        type: binary
      - name: default_profile_image
        type: binary
    #   - name: description
    #     type: text
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
    #   - name: profile_image_path
    #     type: image
        # preprocessing:
        #   num_channels: 3
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

    config = get_default_gbm_model_with_features(config[INPUT_FEATURES], config[OUTPUT_FEATURES])
    from pprint import pprint

    pprint(config)

    LudwigModel(config)

    # TODO: Add an explicit assert for GBM models that none of the features are TEXT or IMAGE.


def test_get_generally_good_model_configs():
    configs = get_generally_good_model_configs()
    print(configs.keys())
