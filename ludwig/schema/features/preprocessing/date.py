from marshmallow_dataclass import dataclass

from ludwig.constants import DATE, MISSING_VALUE_STRATEGY_OPTIONS, PREPROCESSING
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import register_preprocessor
from ludwig.schema.metadata.feature_metadata import FEATURE_METADATA


@register_preprocessor(DATE)
@dataclass(repr=False)
class DatePreprocessingConfig(BasePreprocessingConfig):

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a date column",
        parameter_metadata=FEATURE_METADATA[DATE][PREPROCESSING]["missing_value_strategy"],
    )

    fill_value: str = schema_utils.String(
        default="",
        allow_none=False,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
        parameter_metadata=FEATURE_METADATA[DATE][PREPROCESSING]["fill_value"],
    )

    computed_fill_value: str = schema_utils.String(
        default="",
        allow_none=False,
        description="The internally computed fill value to replace missing values with in case the "
        "missing_value_strategy is fill_with_mode or fill_with_mean",
        parameter_metadata=FEATURE_METADATA[DATE][PREPROCESSING]["computed_fill_value"],
    )

    datetime_format: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="This parameter can either be a datetime format string, or null, in which case the datetime "
        "format will be inferred automatically.",
        parameter_metadata=FEATURE_METADATA[DATE][PREPROCESSING]["datetime_format"],
    )
