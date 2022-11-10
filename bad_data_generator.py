import abc
import argparse
import json
import string
import typing
from copy import deepcopy

import numpy as np
import pandas as pd


class DataModifier(abc.ABC):
    alias = "UNKNOWN"

    _subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._subclasses[cls.__name__] = cls

    @classmethod
    def get_by_alias(cls, name):
        for subclass in cls._subclasses:
            if subclass.alias == name:
                return subclass
        raise KeyError(f"No data modifier for type {name}")

    @abc.abstractmethod
    def __call__(self, value):
        pass

    def roll(self):
        return np.random.random(1)[0]


class StringModifier(DataModifier):

    def __init__(self, whitespaces_chance=0.7, min_whitespaces=1,
                 max_whitespaces=2, chars_replacing_chance=0.3,
                 min_chars_to_replace=1, max_chars_to_replace="all",
                 allowed_chars_to_replace=string.ascii_lowercase):
        self._whitespace_proba = whitespaces_chance
        self._min_whitespaces = min_whitespaces
        self._max_whitespaces = max_whitespaces
        self._chars_replace_proba = chars_replacing_chance
        self._chars_replace_min = min_chars_to_replace
        self._chars_replace_max = max_chars_to_replace
        self._allowed_chars = allowed_chars_to_replace

    def _add_whitespaces(self, value: str):
        min_whitespaces = len(value) if self._min_whitespaces == "all" \
            else self._min_whitespaces
        max_whitespaces = len(value) if self._max_whitespaces == "all" \
            else self._max_whitespaces
        n = min_whitespaces if min_whitespaces == max_whitespaces \
            else np.random.randint(min_whitespaces, max_whitespaces + 1)
        for _ in range(n):
            random_pos = np.random.choice(len(value))
            value = value[:random_pos] + " " + value[random_pos:]
        return value

    def _replace_chars(self, value: str):
        min_chars = len(value) if self._chars_replace_min == "all" \
            else self._chars_replace_min
        max_chars = len(value) if self._chars_replace_max == "all" \
            else self._chars_replace_max
        n = min_chars if min_chars == max_chars \
            else np.random.randint(min_chars, max_chars + 1)

        allowed_chars = list(self._allowed_chars) \
            if isinstance(self._allowed_chars, str) \
            else self._allowed_chars
        for _ in range(n):
            random_pos = np.random.choice(len(value))
            random_char = np.random.choice(allowed_chars)
            value = value[:random_pos] + random_char + value[random_pos + 1:]
        return value

    def __call__(self, value: str):
        result = value
        if self.roll() <= self._whitespace_proba:
            result = self._add_whitespaces(result)
        if self.roll() <= self._chars_replace_proba:
            result = self._replace_chars(result)
        return result


class NumberModifier(DataModifier):

    def __init__(self, number_changing_chance=0.5, adding_chance=0.5,
                 min_number_changing_percent=0.01,
                 max_number_changing_percent=1.):
        self._number_change_proba = number_changing_chance
        self._add_proba = adding_chance
        self._min_num_change_percent = min_number_changing_percent
        self._max_num_change_percent = max_number_changing_percent

    def __call__(self, value: typing.Union[int, float]):
        result = value
        if self.roll() <= self._number_change_proba:
            if self._min_num_change_percent == self._max_num_change_percent:
                change_percent = self._min_num_change_percent
            else:
                change_percent = np.random.randint(
                    self._min_num_change_percent,
                    self._max_num_change_percent + 1)
            val_one_percent = value / 100
            delta = val_one_percent * change_percent
            if self.roll() < self._add_proba:
                result += delta
            else:
                result -= delta
        return result


class BadDataGenerator:
    """
    Generator that modifies tha data by cloning each data row several
    times and applying bad modifications to every cloned row.
    Possible modifications are: adding whitespace in a random place,
    changing characters to any random characters, changing
    numeric data by adding or subtraction a small random value.
    It returns data with original and modified rows.
    """

    def __init__(self, min_additional_rows=1, max_additional_rows=4,
                 adding_whitespace_chance=0.7, min_whitespaces=1,
                 max_whitespaces=3, changing_chars_chance=0.3,
                 min_characters_to_replace=1, max_characters_to_replace=3,
                 allowed_chars_to_replace=string.ascii_lowercase,
                 number_changing_chance=0.5, adding_chance=0.5,
                 min_number_changing_percent=0.01,
                 max_number_changing_percent=1.):
        self._min_add_rows = min_additional_rows
        self._max_add_rows = max_additional_rows
        self._whitespace_proba = adding_whitespace_chance
        self._min_whitespaces = min_whitespaces
        self._max_whitespaces = max_whitespaces
        self._chars_replace_proba = changing_chars_chance
        self._chars_replace_min = min_characters_to_replace
        self._chars_replace_max = max_characters_to_replace
        self._allowed_chars = allowed_chars_to_replace
        self._number_change_proba = number_changing_chance
        self._add_proba = adding_chance
        self._min_num_change_percent = min_number_changing_percent
        self._max_num_change_percent = max_number_changing_percent
        self._data_modifiers = {
            "string": StringModifier(self._whitespace_proba,
                                     self._min_whitespaces,
                                     self._max_whitespaces,
                                     self._chars_replace_proba,
                                     self._chars_replace_min,
                                     self._chars_replace_max,
                                     self._allowed_chars),
            "number": NumberModifier(self._number_change_proba,
                                     self._add_proba,
                                     self._min_num_change_percent,
                                     self._max_num_change_percent)
        }

    def _get_records_to_modify(self, record: dict) -> list[dict]:
        n = np.random.randint(self._min_add_rows, self._max_add_rows + 1)
        return [deepcopy(record) for _ in range(n)]

    def _modify_field(self, value, field_schema):
        if not field_schema:
            return value
        return self._data_modifiers[field_schema["type"]](value)

    def _modify_record(self, record: dict, schema: dict) -> dict:
        return {k: self._modify_field(v, schema.get(k, {}))
                for k, v in record.items()}

    def generate_broken_data(self, data, schema):
        """
        :param list[dict] data: list of data records
        :param dict schema:
        :return: updated data with original and modified rows
        :rtype: list[dict]
        """
        updated_data = []
        for record in data:
            updated_data.append(record)
            cloned_records = self._get_records_to_modify(record)
            updated_records = [self._modify_record(cl, schema)
                               for cl in cloned_records]
            updated_data.extend(updated_records)
        return updated_data


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--file-path", "-f", type=str, required=True,
                           help="path to .csv table with data or "
                                "prepared .json")
    argparser.add_argument("--schema", type=str, required=True,
                           help="path to .json with schema or "
                                "stringified json data")
    argparser.add_argument("--output-path", "-o", type=str,
                           default="spoiled_data.csv",
                           help="path to output .csv table")
    args = argparser.parse_args()
    if args.file_path.endswith(".csv"):
        df = pd.read_csv(args.file_path)
        df = json.loads(df.to_json(orient="records"))
    else:
        with open(args.file_path, encoding="utf-8") as f:
            df = json.load(f)

    if args.schema.startswith("{") or args.schema.startswith("["):
        schema = json.loads(args.schema)
    else:
        with open(args.schema, encoding="utf-8") as f:
            schema = json.load(f)

    data_generator = BadDataGenerator()
    updated_data = data_generator.generate_broken_data(df, schema)
    updated_df = pd.DataFrame(updated_data)
    updated_df.to_csv(args.output_path, index=False)
