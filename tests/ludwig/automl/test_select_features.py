import yaml

from ludwig.automl import select_features

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


def test_get_tabular_features():
    assert select_features.get_tabular_features(MULTI_MODAL_CONFIG["input_features"]) == [
        {"name": "default_profile", "type": "binary"},
        {"name": "default_profile_image", "type": "binary"},
        {"name": "favourites_count", "type": "number"},
        {"name": "followers_count", "type": "number"},
        {"name": "friends_count", "type": "number"},
        {"name": "geo_enabled", "type": "binary"},
        {"name": "lang", "type": "category"},
        {"name": "location", "type": "category"},
        {"name": "profile_background_image_path", "type": "category"},
        {"name": "statuses_count", "type": "number"},
        {"name": "verified", "type": "binary"},
        {"name": "average_tweets_per_day", "type": "number"},
        {"name": "account_age_days", "type": "number"},
    ]


def test_get_tabular_text_features():
    assert select_features.get_tabular_text_features(MULTI_MODAL_CONFIG["input_features"]) == [
        {"name": "default_profile", "type": "binary"},
        {"name": "default_profile_image", "type": "binary"},
        {"name": "description", "type": "text"},
        {"name": "favourites_count", "type": "number"},
        {"name": "followers_count", "type": "number"},
        {"name": "friends_count", "type": "number"},
        {"name": "geo_enabled", "type": "binary"},
        {"name": "lang", "type": "category"},
        {"name": "location", "type": "category"},
        {"name": "profile_background_image_path", "type": "category"},
        {"name": "statuses_count", "type": "number"},
        {"name": "verified", "type": "binary"},
        {"name": "average_tweets_per_day", "type": "number"},
        {"name": "account_age_days", "type": "number"},
    ]


def test_get_text_features():
    assert select_features.get_text_features(MULTI_MODAL_CONFIG["input_features"]) == [
        {"name": "description", "type": "text"},
    ]


def test_get_image_features():
    assert select_features.get_image_features(MULTI_MODAL_CONFIG["input_features"]) == [
        {"name": "profile_image_path", "preprocessing": {"num_channels": 3}, "type": "image"},
    ]


def test_get_tabular_image_features():
    assert select_features.get_tabular_image_features(MULTI_MODAL_CONFIG["input_features"]) == [
        {"name": "default_profile", "type": "binary"},
        {"name": "default_profile_image", "type": "binary"},
        {"name": "favourites_count", "type": "number"},
        {"name": "followers_count", "type": "number"},
        {"name": "friends_count", "type": "number"},
        {"name": "geo_enabled", "type": "binary"},
        {"name": "lang", "type": "category"},
        {"name": "location", "type": "category"},
        {"name": "profile_background_image_path", "type": "category"},
        {"name": "profile_image_path", "type": "image", "preprocessing": {"num_channels": 3}},
        {"name": "statuses_count", "type": "number"},
        {"name": "verified", "type": "binary"},
        {"name": "average_tweets_per_day", "type": "number"},
        {"name": "account_age_days", "type": "number"},
    ]
