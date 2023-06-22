import json
from matplotlib import pyplot as plt
from pathlib import Path
import sys
import os

types = ["bc", "mappo_sp", "ppo_sp", "mappo_bc", "ppo_bc"]
layouts = ["asymmetric_advantages", "coordination_ring", "cramped_room", "forced_coordination"]


def find_file(type, layout, files):
    for file in files:
        if file == f"{layout}_{type}.txt":
            return file
    raise FileNotFoundError(f"{type} and {layout} not found in results")


if __name__ == '__main__':
    results_dir = os.path.join(Path(__file__).parents[0], "results")
    text_files = os.listdir(results_dir)
    data = {}
    for layout in layouts:
        layout_values = {}
        for type in types:
            filename = find_file(type, layout, text_files)
            json_data = open(os.path.join(results_dir, filename))
            parsed_data = json.loads(json_data.read())
            layout_values[type] = parsed_data
        data[layout] = layout_values

    for layout in layouts:
        checkpoints = [int(x[0].split("_")[1]) for x in data[layout][types[1]]]
        fig, ax = plt.subplots()
        for type in types:
            if("bc" == type):
                continue
            curr_values = [x[1][0] for x in data[layout][type]]
            line = ax.plot(checkpoints, curr_values, label=type)
        ax.legend()
        ax.set_title(layout.replace("_", " "))
        ax.set_xlabel("training iteration")
        ax.set_ylabel("reward")
        fig.show()

    for layout in layouts:
        fig, ax = plt.subplots()
        values = []
        errors = []
        for type in types:
            max = 0
            error = 0
            for i in range(len(data[layout][type])):
                curr_value = data[layout][type][i][1][0]
                if(curr_value > max):
                    max = curr_value
                    error = data[layout][type][i][1][1]
            values.append(max)
            errors.append(error)
        ax.set_title(layout.replace("_", " "))
        ax.bar(types, values, yerr = errors)
        ax.set_ylabel("max reward")
        fig.show()