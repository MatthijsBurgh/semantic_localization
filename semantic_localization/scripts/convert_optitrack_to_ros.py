#! /usr/bin/env python
from typing import Any, Dict, List

import csv
import PyKDL as kdl
import yaml

try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper


def read_csv(filename: str) -> List[Dict[str, Any]]:
    """
    Read an Optitrack CSV file and return the data as a list of lists

    :param filename:
    :return:
    """
    with open(filename, "r") as f:
        while True:
            try:
                line = f.readline()
                if line.startswith("Frame"):
                    line = line.replace("Time (Seconds)", "time")
                    line = line.replace("X", "RX", 1)
                    line = line.replace("Y", "RY", 1)
                    line = line.replace("Z", "RZ", 1)
                    line = line.replace("W", "RW", 1)
                    header = line.strip().strip(",").lower().split(",")
                    header.append("error")
                    break
            except StopIteration:
                break

        reader = csv.DictReader(f, fieldnames=header, quoting=csv.QUOTE_NONNUMERIC)
        csv_data = list(reader)
    return csv_data


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


def correct_pose(data: List[Dict[str, Any]], origin: kdl.Frame, ros_corr: kdl.Frame) -> List[Dict[str, Any]]:
    corr = ros_corr * origin.Inverse()
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


def write_yaml(filename: str, data: List[Dict[str, Any]]) -> None:
    """
    Write the data to a YAML file

    :param filename:
    :param data:
    """
    with open(filename, "w") as f:
        yaml.dump(data, f, Dumper=Dumper)


if __name__ == "__main__":
    import argparse
    import glob
    import os.path as path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_directory", default="optitrack", required=False, help="The directory to read the CSV files from"
    )
    parser.add_argument(
        "--origin_filename",
        default=path.join("optitrack", "origin_frame.csv"),
        required=False,
        help="The filename of the origin CSV file to read",
    )

    args = parser.parse_args()

    origin_frame_file: str = args.origin_filename
    origin_data = read_csv(origin_frame_file)
    origin_pose = calculate_mean_pose(origin_data)
    print(f"{origin_pose=}")
    if origin_frame_file.endswith("origin_frame.csv"):
        origin_correction = kdl.Frame.Identity()
        origin_correction.M = kdl.Rotation(kdl.Vector(0, 1, 0), kdl.Vector(-1, 0, 0), kdl.Vector(0, 0, 1)).Inverse()
        origin_pose = origin_pose * origin_correction
        print(f"Corrected:\n{origin_pose=}")

    ros_correction = kdl.Frame.Identity()
    ros_correction.M = kdl.Rotation(kdl.Vector(-1, 0, 0), kdl.Vector(0, 0, 1), kdl.Vector(0, 1, 0))

    data_directory = args.data_directory
    for file in glob.glob(path.join(data_directory, "Take*")):
        output_file = file.lower().replace(" ", "_").replace(".csv", ".yaml")
        print(f"Processing {file=} -> {output_file=}")
        file_data = read_csv(file)
        output_data = correct_pose(file_data, origin_pose, ros_correction)
        print("Writing output file")
        write_yaml(output_file, output_data)
