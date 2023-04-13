#! /usr/bin/env python
from typing import Any, Dict, List, Tuple

import csv
import PyKDL as kdl


def read_csv(filename: str) -> List[Dict[str, Any]]:
    """
    Read converted CSV file and return the data as a list of lists

    :param filename:
    :return:
    """
    with open(filename, "r") as f:
        header = f.readline().strip().split(",")
        reader = csv.DictReader(f, fieldnames=header, quoting=csv.QUOTE_NONNUMERIC)
        csv_data = list(reader)
    return csv_data

# function to split a list of dicts into two lists of dicts based on values being empty or not
def split_data(data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]]]:
    data1 = []
    data2 = []
    for d in data:
        if d["error1"] != "":
            for k in list(d.keys()):
                if k.endswith("2"):
                    d.pop(k)
                if k.endswith("1"):
                    d[k[:-1]] = d.pop(k)
            data1.append(d)
        elif d["error2"] != "":
            for k in list(d.keys()):
                if k.endswith("1"):
                    d.pop(k)
                if k.endswith("2"):
                    d[k[:-1]] = d.pop(k)
            data2.append(d)
        else:
            print(f"Found incorrect entry: {d}")
    return data1, data2

def calculate_mean_pose(data: List[Dict[str, Any]]) -> kdl.Frame:
    x = 0
    y = 0
    z = 0
    yaw = 0
    for r in data:
        pose = kdl.Frame(
            kdl.Rotation.Quaternion(r["rx"], r["ry"], r["rz"], r["rw"]),
            kdl.Vector(r["x"], r["y"], r["z"]),
        )
        x += pose.p.x()
        y += pose.p.y()
        z += pose.p.z()
        yaw += pose.M.GetRPY()[2]
    mean_pose = kdl.Frame(
        kdl.Rotation.RPY(0, 0, yaw / len(data)), kdl.Vector(x / len(data), y / len(data), z / len(data))
    )
    return mean_pose


def correct_pose(data: List[Dict[str, Any]], corr: kdl.Frame) -> List[Dict[str, Any]]:
    for r in data:
        try:
            pose = kdl.Frame(
                kdl.Rotation.Quaternion(r["rx"], r["ry"], r["rz"], r["rw"]),
                kdl.Vector(r["x"], r["y"], r["z"]),
            )
            pose = corr * pose
        except TypeError:
            pose = kdl.Frame.Identity()

        r["x"] = pose.p.x()
        r["y"] = pose.p.y()
        r["z"] = pose.p.z()
        r["rx"], r["ry"], r["rz"], r["rw"] = pose.M.GetQuaternion()
    return data


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
        for i in range(len(data["angle"])):
            writer.writerow({key: data[key][i] for key in data.keys()})


if __name__ == "__main__":
    import argparse
    from copy import deepcopy
    import glob
    from math import pi
    import numpy as np
    import os.path as path
    from statistics import mean, stdev

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_directory", default=".", required=False, help="The directory to read the CSV files from"
    )

    args = parser.parse_args()

    data_directory = args.data_directory
    file = path.join(data_directory, "2023-03-10-16-39-47", "optitrack.csv")
    # file = path.join(data_directory, "2023-03-13-15-17-20", "optitrack.csv")
    output_file = path.join(path.dirname(file), "correction_angles.csv")
    file_data = read_csv(file)
    input_data = convert_list_of_dicts_to_dict_of_lists(file_data)
    input_mean = mean(input_data["z"])
    input_std = stdev(input_data["z"])

    means = []
    stds = []
    angles = np.arange(0, 15.25, 0.25)
    for angle in angles:
        correction = kdl.Frame.Identity()  # type: kdl.Frame
        correction.M.DoRotY(angle/180*pi)
        print(f"Processing {file=}\n{angle=}")
        output_data = correct_pose(deepcopy(file_data), correction.Inverse())
        output_data = convert_list_of_dicts_to_dict_of_lists(output_data)
        output_mean = mean(output_data["z"])
        output_std = stdev(output_data["z"])
        means.append(output_mean)
        stds.append(output_std)
        print(f"Input mean: {input_mean} std: {input_std}")
        print(f"Output mean: {output_mean} std: {output_std}")
        # import matplotlib.pyplot as plt
        # plt.plot(input_data["z"], label="input_z")
        # plt.plot(output_data["z"], label="output_z")
        # plt.legend()
        # plt.title(f"Angle: {angle},\nInput mean: {input_mean} std: {input_std},\nOutput mean: {output_mean} std: {output_std}")
        # plt.show()

    write_csv(output_file, {"angle": angles, "mean": means, "std": stds})
