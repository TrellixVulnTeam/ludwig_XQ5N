#! /usr/bin/env python
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import copy
import functools
import os
import random
import subprocess
import weakref
from collections import OrderedDict
from collections.abc import Mapping

import numpy
import torch

from ludwig.constants import ENCODER, NAME, PROC_COLUMN
from ludwig.globals import DESCRIPTION_FILE_NAME
from ludwig.utils import fs_utils
from ludwig.utils.fs_utils import find_non_existing_dir_by_adding_suffix


def set_random_seed(random_seed):
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    random.seed(random_seed)
    numpy.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)


def merge_dict(dct, merge_dct):
    """Recursive dict merge. Inspired by :meth:``dict.update()``, instead of updating only top-level keys,
    dict_merge recurses down into dicts nested to an arbitrary depth, updating keys. The ``merge_dct`` is merged
    into ``dct``.

    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    dct = copy.deepcopy(dct)
    for k, v in merge_dct.items():
        if k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], Mapping):
            dct[k] = merge_dict(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]
    return dct


def sum_dicts(dicts, dict_type=dict):
    summed_dict = dict_type()
    for d in dicts:
        for key, value in d.items():
            if key in summed_dict:
                prev_value = summed_dict[key]
                if isinstance(value, (dict, OrderedDict)):
                    summed_dict[key] = sum_dicts([prev_value, value], dict_type=type(value))
                elif isinstance(value, numpy.ndarray):
                    summed_dict[key] = numpy.concatenate((prev_value, value))
                else:
                    summed_dict[key] = prev_value + value
            else:
                summed_dict[key] = value
    return summed_dict


def resolve_pointers(dict1, dict2, dict2_name):
    resolved_dict = copy.deepcopy(dict1)
    for key in dict1:
        value = dict1[key]
        if value.startswith(dict2_name):
            key_in_dict2 = value[len(dict2_name) :]
            if key_in_dict2 in dict2.keys():
                value = dict2[key_in_dict2]
                resolved_dict[key] = value
            if ENCODER in dict2.keys():  # TODO(Connor): Temporary fix, remove this during preproc refactor
                if key_in_dict2 in dict2[ENCODER].keys():
                    value = dict2[ENCODER][key_in_dict2]
                    resolved_dict[key] = value
    return resolved_dict


def get_from_registry(key, registry):
    if hasattr(key, "lower"):
        key = key.lower()
    if key in registry:
        return registry[key]
    else:
        raise ValueError(f"Key {key} not supported, available options: {registry.keys()}")


def set_default_value(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = value


def set_default_values(dictionary: dict, default_value_dictionary: dict):
    """This function sets multiple default values recursively for various areas of the config. By using the helper
    function set_default_value, It parses input values that contain nested dictionaries, only setting values for
    parameters that have not already been defined by the user.

    Args:
        dictionary (dict): The dictionary to set default values for, generally a section of the config.
        default_value_dictionary (dict): The dictionary containing the default values for the config.
    """
    for key, value in default_value_dictionary.items():
        if key not in dictionary:  # Event where the key is not in the dictionary yet
            dictionary[key] = value
        elif value == {}:  # Event where dict is empty
            set_default_value(dictionary, key, value)
        elif isinstance(value, dict) and value:  # Event where dictionary is nested - recursive call
            set_default_values(dictionary[key], value)
        else:
            set_default_value(dictionary, key, value)


def get_class_attributes(c):
    return {i for i in dir(c) if not callable(getattr(c, i)) and not i.startswith("_")}


def get_output_directory(output_directory, experiment_name, model_name="run"):
    base_dir_name = os.path.join(output_directory, experiment_name + ("_" if model_name else "") + (model_name or ""))
    return fs_utils.abspath(find_non_existing_dir_by_adding_suffix(base_dir_name))


def get_file_names(output_directory):
    description_fn = os.path.join(output_directory, DESCRIPTION_FILE_NAME)
    training_stats_fn = os.path.join(output_directory, "training_statistics.json")

    model_dir = os.path.join(output_directory, "model")

    return description_fn, training_stats_fn, model_dir


def get_combined_features(config):
    return config["input_features"] + config["output_features"]


def get_proc_features(config):
    return get_proc_features_from_lists(config["input_features"], config["output_features"])


def get_proc_features_from_lists(*args):
    return {feature[PROC_COLUMN]: feature for features in args for feature in features}


def set_saved_weights_in_checkpoint_flag(config_obj):
    """Adds a flag to all input feature encoder configs indicating that the weights are saved in the checkpoint.

    Next time the model is loaded we will restore pre-trained encoder weights from ludwig model (and not load from cache
    or model hub).
    """
    for input_feature in config_obj.input_features.to_list():
        input_feature_name = input_feature[NAME]
        encoder_obj = config_obj.input_features.get(input_feature_name).encoder
        encoder_obj.saved_weights_in_checkpoint = True


def remove_empty_lines(str):
    return "\n".join([line.rstrip() for line in str.split("\n") if line.rstrip()])


# TODO(travis): move to cached_property when we drop Python 3.7.
# https://stackoverflow.com/a/33672499
def memoized_method(*lru_args, **lru_kwargs):
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(self, *args, **kwargs):
            # We're storing the wrapped method inside the instance. If we had
            # a strong reference to self the instance would never die.
            self_weak = weakref.ref(self)

            @functools.wraps(func)
            @functools.lru_cache(*lru_args, **lru_kwargs)
            def cached_method(*args, **kwargs):
                return func(self_weak(), *args, **kwargs)

            setattr(self, func.__name__, cached_method)
            return cached_method(*args, **kwargs)

        return wrapped_func

    return decorator


def get_commit_hash():
    """If Ludwig is run from a git repository, get the commit hash of the current HEAD.

    Returns None if git is not executable in the current environment or Ludwig is not run in a git repo.
    """
    try:
        with open(os.devnull, "w") as devnull:
            is_a_git_repo = subprocess.call(["git", "branch"], stderr=subprocess.STDOUT, stdout=devnull) == 0
        if is_a_git_repo:
            commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8")
            return commit_hash
    except:  # noqa: E722
        pass
    return None
