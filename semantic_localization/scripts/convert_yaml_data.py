#! /usr/bin/env python
from typing import Any, Dict, List

import yaml
import csv

try:
    from yaml import CDumper as Dumper, CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader


def read_yaml(filename: str) -> List[Dict[str, Any]]:
    """
    Read the data from a YAML file

    :param filename:
    :return:
    """
    with open(filename, "r") as f:
        data = yaml.load(f, Loader=Loader)
    return data


def write_yaml(filename: str, data: List[Dict[str, Any]]) -> None:
    """
    Write the data to a YAML file

    :param filename:
    :param data:
    """
    with open(filename, "w") as f:
        yaml.dump(data, f, Dumper=Dumper)


# function to write a dict of lists to a csv file
def write_csv(filename: str, data: Dict[str, List[Any]]) -> None:
    """
    Write the data to a CSV file

    :param filename:
    :param data:
    """
    with open(filename, "w") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        writer.writeheader()
        for i in range(len(data["x"])):
            writer.writerow({key: data[key][i] for key in data.keys()})


# function to eliminate data when error is empty
def filter_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for dictionary in data:
        if dictionary["error"] != "":
            output.append(dictionary)
    return output


# function to convert a list of dictionaries to a dictionary of lists
def convert_list_of_dicts_to_dict_of_lists(data: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    # get the keys from the first dictionary
    keys = data[0].keys()
    # create a dictionary of lists
    dict_of_lists = {key: [] for key in keys}
    # iterate over the list of dictionaries
    for dictionary in data:
        # iterate over the keys
        for key in keys:
            # append the value of the key to the list
            dict_of_lists[key].append(dictionary[key])
    return dict_of_lists


if __name__ == "__main__":
    import argparse
    import glob
    import os.path as path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_directory", default="optitrack", required=False, help="The directory to read the CSV files from"
    )

    args = parser.parse_args()

    data_directory = args.data_directory
    for file in glob.glob(path.join(data_directory, "**", "optitrack.yaml"), recursive=True):
        output_file = file.replace(".yaml", ".csv")
        print(f"Processing {file=} -> {output_file=}")
        file_data = read_yaml(file)
        filtered_data = filter_data(file_data)
        output_data = convert_list_of_dicts_to_dict_of_lists(filtered_data)
        print("Writing output file")
        write_csv(output_file, output_data)
