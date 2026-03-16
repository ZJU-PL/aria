import argparse
import os
import json
import csv
import seaborn as sns
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
from collections import defaultdict, Counter
import re


def is_isolated_word(word, text):
    return re.search(rf"\b{word}\b", text) is not None


all_solver = ["z3", "cvc5", "yices2", "bitwuzla"]
all_bit_blasting_solver = ["caqe", "q3b", "cd15", "lgl", "mc"]
all_cegis_solver = ["cegis_z3", "cegis_msat", "cegis_yices", "cegis_btor", "cegis_cvc4"]
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
SAVEDIR = CUR_DIR + "/Result"
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["axes.labelsize"] = 12


class Experiment_Helper:
    def __init__(self, data_list) -> None:
        self.data = data_list

    def generate_row_elements(self, entry):
        row_elements = []

        if "smt_solver" in entry and entry["smt_solver"] != "cegis":
            row_elements.append(entry["smt_solver"])
        elif "cegis_solver" in entry:
            row_elements.append("cegis_" + entry["cegis_solver"])
        else:
            row_elements.append("")

        if "method" in entry:
            row_elements.append(entry["method"])
        else:
            row_elements.append("")

        if "template" in entry:
            row_elements.append(entry["template"])
        else:
            row_elements.append("")

        if "strengthen" in entry:
            row_elements.append("True")
        else:
            row_elements.append("False")

        if "num_disjunctions" in entry:
            row_elements.append(entry["num_disjunctions"])
        else:
            row_elements.append(1)

        return row_elements

    def generate_table(self, output_file):
        config_data = defaultdict(lambda: defaultdict(Counter))

        # Predefined configurations
        disjunction_config = {None: "Base", 2: "DIS_2", 5: "DIS_5", 10: "DIS_10"}

        # Traverse through the data and count the results for each configuration
        for entry in self.data:
            method = entry.get("method")
            smt_solver = entry.get("smt_solver")
            cegis_solver = entry.get("cegis_solver")

            if method == "efsmt" and smt_solver == "z3":
                label = "QI-Z3"
            elif method == "efsmt" and smt_solver == "q3b":
                label = "BB-Q3B"
            elif smt_solver == "cegis" and cegis_solver == "yices":
                label = "IS-Yices"
            else:
                continue

            num_disj = entry.get("num_disjunctions")
            strengthen = entry.get("strengthen", False)

            # Determine the configuration
            base_config = disjunction_config.get(num_disj, "N/A")
            if strengthen and base_config == "Base":
                config_name = "PS"
            elif strengthen:
                config_name = base_config + "+PS"
            else:
                config_name = base_config
            result = ""
            if entry.get("safe") == True:
                result = "T"
            elif entry.get("timeout") == True:
                result = "X"
            elif entry.get("unknown") == True:
                result = "U"
            elif entry.get("unexpected_error") == True:
                result = "Err"
            else:
                result = "N/A"

            template = entry.get("template", "")
            if "power_" in template:
                template = str(template).replace("power_", "")
            if template in ["bv_interval", "bv_octagon", "bv_poly", "bv_zone"]:
                config_data[(label, config_name)][template][result] += 1
            else:
                print(template)
                exit(0)

        # Write the table to a CSV file
        with open(output_file, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)

            # Write header
            header = [
                "Solver",
                "Configuration",
                "bv_interval",
                "bv_octagon",
                "bv_poly",
                "bv_zone",
            ]
            csvwriter.writerow(header)

            # Write rows
            for (solver, config), templates_data in config_data.items():
                row = [solver, config]
                for template in ["bv_interval", "bv_octagon", "bv_poly", "bv_zone"]:
                    data = templates_data[template]
                    row_data = "T: {T}, U: {U}, Others: {Others}".format(
                        T=data["T"],
                        U=data["U"],
                        Others=data["F"] + data["Err"] + data["X"] + data.get("N/A", 0),
                    )
                    row.append(row_data)
                csvwriter.writerow(row)

    def generate_csv(self, output_filename):
        file_data = defaultdict(dict)
        stats = defaultdict(Counter)
        times = defaultdict(float)
        benchmark_counts = defaultdict(int)
        for entry in self.data:
            row_elements = self.generate_row_elements(entry)
            row_key = tuple(row_elements)
            result = ""
            if entry.get("safe") == True:
                result = "T"
            elif entry.get("safe") == False:
                result = "F"
            elif entry.get("timeout") == True:
                result = "X"
            elif entry.get("unknown") == True:
                result = "U"
            elif entry.get("unexpected_error") == True:
                result = "Err"
            else:
                result = "N/A"
            file_data[row_key][entry["file"]] = result
            stats[row_key][result] += 1
            if "time" in entry:
                times[row_key] += entry["time"]
                benchmark_counts[row_key] += 1

        with open(output_filename, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)

            # Collect column names
            column_names = set()
            for results in file_data.values():
                column_names.update(results.keys())
            sorted_column_names = sorted(list(column_names))

            # Write header
            header = [
                "SMT Solver",
                "Method",
                "Template",
                "Strengthen",
                "Num Disjunctions",
            ] + sorted_column_names
            csvwriter.writerow(header)

            # Write rows
            for row_key, results in file_data.items():
                row = list(row_key)
                for column_name in sorted_column_names:
                    row.append(results.get(column_name, "N/A"))
                csvwriter.writerow(row)

            csvwriter.writerow([])
            stats_header = [
                "SMT Solver",
                "Method",
                "Template",
                "Strengthen",
                "Num Disjunctions",
                "T",
                "F",
                "Err",
                "X",
                "U",
                "N/A",
                "Avg Time",
            ]
            csvwriter.writerow(stats_header)
            for row_key, counter in stats.items():
                stats_row = list(row_key)
                for key in stats_header[5:]:
                    stats_row.append(counter[key])
                avg_time = (
                    times[row_key] / benchmark_counts[row_key]
                    if benchmark_counts[row_key] > 0
                    else 0
                )
                print(benchmark_counts[row_key])
                stats_row.append(avg_time)
                csvwriter.writerow(stats_row)

    def calculate_verified_rate(self):
        result_df = pd.DataFrame(columns=["Tool", "Template", "Verified Rate"])

        templates = ["bv_interval", "bv_zone", "bv_octagon", "bv_poly"]
        solvers = [("z3", None), ("q3b", None), ("cegis", "yices")]
        template_map = {
            "bv_interval": "int",
            "bv_zone": "zone",
            "bv_octagon": "oct",
            "bv_poly": "poly",
        }
        result = {}
        for entry in self.data:
            if (
                "num_disjunctions" in entry
                or "power" in entry.get("template", "")
                or "strength" in entry
                or "template" not in entry
            ):
                continue

            template = entry["template"]
            if template not in templates:
                continue
            template = template_map[template]
            smt_solver = entry.get("smt_solver")
            cegis_solver = entry.get("cegis_solver")
            solver_tuple = (smt_solver, cegis_solver)

            if solver_tuple not in solvers:
                continue

            safe = entry.get("safe", False)

            key = (smt_solver, template)
            if key not in result:
                result[key] = {"correct": 0, "total": 0}

            result[key]["total"] += 1
            if safe:
                result[key]["correct"] += 1

        for key, value in result.items():
            solver, template = key
            rate = value["correct"] / value["total"] * 100
            new_row = pd.DataFrame(
                [{"Tool": solver, "Template": template, "Verified Rate": rate}]
            )
            result_df = pd.concat([result_df, new_row], ignore_index=True)

        return result_df

    def plot_grouped_bar_chart(self):
        verified_rate_df = self.calculate_verified_rate()

        color_dict = {"z3": "#9dc3e7", "q3b": "#a9d18f", "cegis": "#e5b1a9"}

        sns.set_style("whitegrid")

        # 创建一个颜色列表，用于barplot中的palette参数
        color_palette = [color_dict[tool] for tool in verified_rate_df["Tool"].unique()]

        ax = sns.barplot(
            data=verified_rate_df,
            x="Template",
            y="Verified Rate",
            hue="Tool",
            palette=color_palette,
        )

        # 设置填充样式
        hatches = ["", "/", "x"]
        num_tools = len(verified_rate_df["Tool"].unique())

        # 对于每个工具，设置相应的填充样式
        for i, tool in enumerate(verified_rate_df["Tool"].unique()):
            bars = [
                patch
                for patch in ax.patches
                if patch.get_fc() == ax.get_legend().legend_handles[i].get_facecolor()
            ]
            for bar in bars:
                bar.set_hatch(hatches[i])

        # 重命名图例标签
        new_legend_labels = {"z3": "QI_z3", "q3b": "BB_q3b", "cegis": "IS_yices"}

        handles, labels = ax.get_legend_handles_labels()
        font_properties = FontProperties(weight="bold")
        ax.legend(
            handles,
            [new_legend_labels[label] for label in labels],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.10),
            ncol=len(verified_rate_df["Tool"].unique()),
            frameon=False,
            prop=font_properties,
        )
        plt.tight_layout()

        plt.savefig("Verified rate.pdf")

    def plot_cumulative_time_per_template(self, output_file):
        templates = ["bv_poly", "bv_interval", "bv_zone", "bv_octagon"]

        # Adjusted colors to be lighter and more muted
        colors = {
            "bv_poly": "#FF6666",  # Lighter Red
            "bv_interval": "#66B266",  # Lighter Green
            "bv_zone": "#FFCC99",  # Lighter Orange
            "bv_octagon": "#FFFF99",  # Lighter Yellow
        }
        label_mapping = {
            "bv_poly": "poly/QI_z3",
            "bv_interval": "int/QI_z3",
            "bv_zone": "zone/QI_z3",
            "bv_octagon": "oct/QI_z3",
        }

        # Line styles for each template
        line_styles = {
            "bv_poly": ":",
            "bv_interval": "--",
            "bv_zone": "-.",
            "bv_octagon": "-",
        }

        cumulative_data = {template: [] for template in templates}
        time_data = {template: [] for template in templates}

        for data_entry in self.data:
            if (
                data_entry.get("template", "") in templates
                and data_entry["smt_solver"] == "z3"
                and data_entry["lang"] == "chc"
                and data_entry["method"] == "efsmt"
            ):

                cumulative_data[data_entry["template"]].append(data_entry["time"])

        for template, times in cumulative_data.items():
            cumulative_time = 0
            for time in times:
                cumulative_time += time
                time_data[template].append(cumulative_time)
        plt.figure()
        for template, times in time_data.items():
            sns.lineplot(
                x=list(range(len(times))),
                y=times,
                label=label_mapping[template],
                color=colors[template],
                linestyle=line_styles[template],
            )

        plt.title("Cumulative Time vs. Benchmark Index")
        plt.xlabel("Benchmark Index")
        plt.ylabel("Cumulative Time (s)")

        # Setting the plot background color to light gray
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_edgecolor("#D0D0D0")  # Light gray

        # Adjusting the tick color to match the spine color
        ax.tick_params(colors="#D0D0D0")
        # Bold legend text
        font_properties = FontProperties(weight="bold")
        plt.legend(
            fontsize="small",
            prop=font_properties,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=4,
            frameon=False,
        )

        plt.savefig("./cumulative_time_per_template.pdf", dpi=600, format="pdf")

    def heat_map_between_teamplte_and_solver(self, output_file="."):
        method_data = []
        NameMap = {""}
        for entry in self.data:
            if entry["method"] == "efsmt":
                result = {
                    "solver": (
                        entry["smt_solver"]
                        if entry["smt_solver"] != "cegis"
                        else entry["cegis_solver"]
                    ),
                    "template": entry["template"],
                    "time": entry["time"],
                    "success": entry.get("safe", False),
                }
                if result["solver"] in all_solver:
                    result["solver"] = "QI_" + result["solver"]
                elif result["solver"] in all_bit_blasting_solver:
                    result["solver"] = "BB_" + result["solver"]
                elif "cegis_" + result["solver"] in all_cegis_solver:
                    result["solver"] = "IS_" + result["solver"]
                method_data.append(result)
        template_map = {
            "bv_interval": "int",
            "bv_zone": "zone",
            "bv_octagon": "oct",
            "bv_poly": "poly",
            "power_bv_interval": "PW_int",
            "power_bv_zone": "PW_zone",
            "power_bv_octagon": "PW_oct",
            "power_bv_poly": "PW_poly",
        }
        heatmap_data = {}
        for entry in method_data:
            template = entry["template"]
            key = (entry["solver"], template_map[template])
            if key not in heatmap_data:
                heatmap_data[key] = {
                    "success_count": 1 if entry["success"] else 0,
                    "total_count": 1,
                }
            else:
                heatmap_data[key]["total_count"] += 1
                if entry["success"]:
                    heatmap_data[key]["success_count"] += 1

        success_rates = {
            key: value["success_count"] / value["total_count"]
            for key, value in heatmap_data.items()
        }
        heatmap_df = pd.DataFrame.from_dict(
            success_rates, orient="index", columns=["success_rate"]
        )
        heatmap_df = heatmap_df.reset_index()
        heatmap_df[["solver", "template"]] = pd.DataFrame(
            heatmap_df["index"].tolist(), index=heatmap_df.index
        )
        heatmap_df = heatmap_df.pivot_table(
            values="success_rate", index=["solver"], columns=["template"]
        )

        plt.figure(figsize=(8, 6))
        sns.set(font_scale=0.8)  # Adjust font scale
        # cmap = sns.light_palette(as_cmap=True, start=.5, rot=-.75)  # Use a softer color palette
        sns.heatmap(
            heatmap_df,
            annot=True,
            cmap="Blues",
            fmt=".1%",
            linewidths=0.5,
            linecolor="gray",
        )  # Add linewidths and linecolor
        plt.title("Verified Rate Heatmap (only PS)")
        plt.xlabel("Template")
        plt.ylabel("Solver")

        file_name = "./verfied_heatmap.pdf"
        plt.savefig(file_name, dpi=300, format="pdf")

        time_data = {}
        for entry in method_data:
            template = entry["template"]
            key = (entry["solver"], template_map[template])
            if key not in time_data:
                time_data[key] = {"total_time": entry["time"], "count": 1}
            else:
                time_data[key]["total_time"] += entry["time"]
                time_data[key]["count"] += 1

        average_times = {
            key: value["total_time"] / value["count"] if value["count"] > 0 else 0
            for key, value in time_data.items()
        }
        for key, value in time_data.items():
            print(value["count"])
        time_heatmap_df = pd.DataFrame.from_dict(
            average_times, orient="index", columns=["average_time"]
        )
        time_heatmap_df = time_heatmap_df.reset_index()
        time_heatmap_df[["solver", "template"]] = pd.DataFrame(
            time_heatmap_df["index"].tolist(), index=time_heatmap_df.index
        )
        time_heatmap_df = time_heatmap_df.pivot_table(
            values="average_time", index=["solver"], columns=["template"]
        )

        plt.figure(figsize=(8, 6))
        sns.set(font_scale=0.8)
        sns.heatmap(
            time_heatmap_df,
            annot=True,
            cmap="Blues",
            fmt=".1f",
            linewidths=0.5,
            linecolor="gray",
        )
        plt.title("Average Time Heatmap (only PS)")
        plt.xlabel("Template")
        plt.ylabel("Solver")

        file_name_time = "./average_time_heatmap.pdf"
        plt.savefig(file_name_time, dpi=300, format="pdf")

    def box_map(self, output_file):
        boxplot_data = []
        for entry in self.data:
            if entry["method"] == "efsmt":
                result = {
                    "solver": (
                        entry["smt_solver"]
                        if entry["smt_solver"] != "cegis"
                        else entry["cegis_solver"]
                    ),
                    "template": entry["template"],
                    "time": entry["time"],
                    "success": entry.get("safe", False),
                }
                boxplot_data.append(result)

        # 将数据转换为 DataFrame
        boxplot_df = pd.DataFrame(boxplot_data)

        # 绘制箱线图
        plt.figure()
        sns.boxplot(data=boxplot_df, x="template", y="time", hue="solver")
        plt.title("Execution Time Box Plot (Method: efsmt)")
        plt.xlabel("Template")
        plt.ylabel("Time (s)")
        plt.savefig(os.path.join(output_file, "boxplot.png"), dpi=300)

    def prop_strengthen_compare(self, output_file):
        efsmt_data = [entry for entry in self.data if entry["method"] == "efsmt"]

        # 计算不同组合成功率
        success_rate_data = {}
        for entry in efsmt_data:
            key = (
                (
                    entry["smt_solver"]
                    if entry["smt_solver"] != "cegis"
                    else entry["cegis_solver"]
                ),
                entry["template"],
                entry.get("strengthen", False),
            )
            if key not in success_rate_data:
                success_rate_data[key] = {
                    "success_count": 1 if entry.get("safe") else 0,
                    "total_count": 1,
                }
            else:
                success_rate_data[key]["total_count"] += 1
                if entry.get("safe"):
                    success_rate_data[key]["success_count"] += 1

        # 将成功率数据转换为 Pandas DataFrame
        success_rate_df = pd.DataFrame.from_dict(
            success_rate_data, orient="index", columns=["success_count", "total_count"]
        )
        # 添加这一行以创建名为 'index' 的新列
        success_rate_df["index"] = success_rate_df.index
        success_rate_df = success_rate_df.reset_index(
            drop=True
        )  # 使用 drop=True 以避免在重置索引时创建额外的列
        success_rate_df[["solver", "template", "strengthen"]] = pd.DataFrame(
            success_rate_df["index"].tolist(), index=success_rate_df.index
        )
        success_rate_df["success_rate"] = (
            success_rate_df["success_count"] / success_rate_df["total_count"]
        )
        success_rate_df["solver_template"] = (
            success_rate_df["solver"] + " - " + success_rate_df["template"]
        )

        # 创建分组条形图
        plt.figure(figsize=(8, 6))
        sns.set(font_scale=0.8)  # Adjust font scale
        sns.barplot(
            x="solver_template",
            y="success_rate",
            hue="strengthen",
            data=success_rate_df,
            palette="RdYlBu",
        )
        plt.title(
            "Success Rate by Solver-Template Combinations and Strengthen Parameter"
        )
        plt.xlabel("Solver - Template Combination")
        plt.ylabel("Success Rate")
        plt.xticks(rotation=45)

        plt.savefig(
            os.path.join(output_file, "grouped_barplot_success_rate.png"),
            dpi=300,
            bbox_inches="tight",
        )


def init(mode):
    data_list = []
    for subdir, dirs, files in os.walk(SAVEDIR):

        for file in files:
            print(file)
            if file.endswith(".json"):
                file_path = os.path.join(subdir, file)
                with open(file_path, "r") as json_file:
                    data = json.load(json_file)
            else:
                continue
            if 1 in mode:
                messages = data.get("message", [])
                if not messages:
                    os.remove(file_path)
                    continue
                all_empty_or_traceback = all(not m.strip() for m in messages) or any(
                    "Traceback" in m for m in messages
                )
                data["file"] = os.path.splitext(data["file"])[0]
                if all_empty_or_traceback:
                    if data.get("unexpected_error", False) != True:
                        print(f"Unexpected error not found in file: {file_path}")
                        data["unexpected_error"] = True
                    data.pop("unknown", None)
                    data.pop("safe", None)
                else:
                    if any(
                        is_isolated_word("safe", m) or "define-fun inv_" in m
                        for m in messages
                    ):
                        data["safe"] = True
                    elif any(
                        is_isolated_word("infeasible", m)
                        or is_isolated_word("unsat", m)
                        for m in messages
                    ):
                        data["safe"] = False
                    elif any(is_isolated_word("sat", m) for m in messages):
                        data["safe"] = True
                    elif any(is_isolated_word("unknown", m) for m in messages):
                        data["unknown"] = True
                    data.pop("unexpected_error", None)
                    if "unknown" not in data:
                        data.pop("unknown", None)
                    else:
                        data.pop("safe", None)

                with open(file_path, "w") as json_file:
                    json.dump(data, json_file, indent=4)
            if 11 in mode:
                # if 'num_disjunctions' in data and data['num_disjunctions'] == 2:
                # if 'smt_solver' not in data or data['smt_solver'] == 'g4' or data['smt_solver'] == 'gc4':
                if data["lang"] == "sygus" and "strengthen" in data:
                    # if 'num_disjunctions' not in data and 'power' in data.get('template', ''):
                    os.remove(file_path)
                    print(f"{file_path} has been deleted.")
            if 13 in mode:
                if data["lang"] == "sygus":
                    filename = data["file"]
                    index = filename.find(".smt2")
                    if index != -1:
                        new_filename = filename[: index + len(".smt2")]
                    else:
                        print(data["file"])
                    data["file"] = new_filename
                    with open(file_path, "w") as json_file:
                        json.dump(data, json_file, indent=4)

            if "bits" not in data:
                bits_pattern = re.compile(r"(\d+bits_[un]*signed)")
                bits_str = bits_pattern.search(file_path).group()
                # print(bits_str)
                data["bits"] = bits_str
                if bits_str not in data["file"]:
                    data["file"] = data["file"] + "_" + bits_str
            if "unsigned" in data["bits"]:
                data_list.append(data)
    safe_file = set()
    delete_file = set()
    if 12 in mode:
        num_disjunctions_counts = {2: 0, 5: 0, 10: 0}
        method_counts = {"pdr": 0, "eld": 0}

        for data in data_list:
            if "num_disjunctions" in data:
                if data["num_disjunctions"] in num_disjunctions_counts:
                    num_disjunctions_counts[data["num_disjunctions"]] += 1
            if "method" in data and data["method"] in method_counts:
                method_counts[data["method"]] += 1
        print(num_disjunctions_counts, method_counts)
    for data in data_list:
        if data.get("safe", "") == True:
            safe_file.add(data["file"])
        elif data.get("safe", "") == False:
            delete_file.add(data["file"])
    for file in delete_file:
        if file in safe_file:
            safe_file.remove(file)
    return data_list, list(safe_file)


def collect_data(chc_dir, strengthen=False, disjunction=0):
    data = {}
    for root, _, files in os.walk(chc_dir):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), "r") as f:
                    content = json.load(f)
                    if (
                        "unsigned" in content["bits"]
                        and content["method"] == "efsmt"
                        and content.get("strengthen", False) == strengthen
                        and content.get("num_disjunctions", 0) == disjunction
                    ):
                        template = content["template"]
                        bits = content["bits"]
                        source = root.split("/")[-3]
                        # print(source)
                        solver = content["smt_solver"]
                        if solver == "cegis":
                            solver = "cegis" + content["cegis_solver"]
                        if bits not in data:
                            data[bits] = {}
                        if source not in data[bits]:
                            data[bits][source] = {}
                        if template not in data[bits][source]:
                            data[bits][source][template] = {}
                        if solver not in data[bits][source][template]:
                            data[bits][source][template][solver] = {
                                "safe": 0,
                                "unknown": 0,
                                "timeout": 0,
                            }
                        if "safe" in content:
                            data[bits][source][template][solver]["safe"] += content[
                                "safe"
                            ]
                        elif "timeout" in content:
                            data[bits][source][template][solver]["timeout"] += content[
                                "timeout"
                            ]
                        elif "unknown" in content:
                            data[bits][source][template][solver]["unknown"] += content[
                                "unknown"
                            ]
                        else:
                            data[bits][source][template][solver]["unknown"] += 1
                        # if solver == 'z3' and template == 'bv_octagon' and source == '2022.multi-phase':
                        #     print(file, count)
                        #     count += 1
                        #     print(data[bits][source][template][solver])
    return data


def generate_csv(data, output_file="rq1_basic_four_template_compare.csv"):
    with open(output_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ["Bits", "Source", "Template", "Solver", "Safe", "Unknown", "Timeout"]
        csv_writer.writerow(header)

        solver_counts = Counter()
        for bit in data:
            for source in data[bit]:
                for template in data[bit][source]:
                    solvers = data[bit][source][template].keys()
                    for solver in solvers:
                        count = data[bit][source][template][solver]["safe"]
                        solver_counts[solver] += count

        top_solvers = solver_counts.most_common(3)
        top_solver_names = [solver for solver, _ in top_solvers]

        for bit in data:
            for source in data[bit]:
                for template in data[bit][source]:
                    for solver in top_solver_names:
                        if solver in data[bit][source][template]:
                            row = [bit, source, template, solver]
                            for stat in ["safe", "unknown", "timeout"]:
                                count = data[bit][source][template][solver][stat]
                                row.append(count)
                            csv_writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process command line arguments")
    parser.add_argument("--mode", nargs="+", type=int, help="one or more mode numbers")
    parser.add_argument("--disjunction", type=int)
    # 1 -> with initial
    # 11 -> temporary process files
    # 2 -> start experiment and paint the basic data without property strengthen and disjunctions.
    # 3 -> final experiment for solution table:
    args = parser.parse_args()
    data_list, safe_file = init(args.mode)
    data_list = [data for data in data_list if data["file"] in safe_file]
    print("-------------finish init data------------------------")

    if 2 in args.mode:
        # new_data_list = [
        #     data for data in data_list if not data.get('strengthen')]
        RQ1 = Experiment_Helper(data_list)
        if not os.path.exists(SAVEDIR):
            os.makedirs(SAVEDIR)
        if 21 in args.mode:
            RQ1.generate_csv(os.path.join(SAVEDIR, "data.csv"))
            # RQ1.plot_cumulative_time_per_template(SAVEDIR)
            # RQ1.variouts_images(SAVEDIR)
        if 22 in args.mode:
            RQ1.plot_grouped_bar_chart()
    if 3 in args.mode:
        RQ13 = Experiment_Helper(data_list)
        RQ13.generate_table(output_file="table.csv")
        # data = collect_data('./Result/chc')
    if 4 in args.mode:
        data_list = [data for data in data_list if "strengthen" in data]
        RQ14 = Experiment_Helper(data_list)
        RQ14.heat_map_between_teamplte_and_solver()

    if 6 in args.mode:
        choosen_method = ["eld", "pdr", "cvc5sys"]
        choosen_template = ["none"]
        data_list = [
            data
            for data in data_list
            if data["method"] in choosen_method
            or (data["method"] == "efsmt" and data["template"] == choosen_method)
        ]
        RQ2 = Experiment_Helper()
    # DONE:efsmt方法中，不同模板，求解器组合对应的'safe'验证成功率 【对应热力学图】
    # DONE:所有可选参数组合的验证情况【csv表格】
    # DONE:验证成功率最高的十种方法组合对应的时间累积图，正确率累积图【折线图】
    # DOING:将时间延长至600s，然后重跑上述实验+绘图
    # TODO:对比strengthen有无的结果，由于所有的组合太多，都花在一张图中的效果非常不好，因此需要specify 'z3' solver，有无strengthen的结果。【分组柱状图】
    # TODO:关于disjunctions数量变化，efsmt求解的正确率变化【折线图】
    # DONE:不同类型的图像？比如小提琴图，箱线图等等 【由于排列组合太多，箱线图只能针对一部分数据组合进行绘制，但箱线图可以观察时间的总体趋势】
    # DONE:指定某个solver，method，template，strengthen，num_disjunctions，观察其余最高

    # 2. compare different template and different strategy of template method. [no disjunctions and strengthen]
    #    (1) bv_interval 模板表达式的溢出带来的精度丢失，反而降低了其他模板的精度。
    #    (2) 复杂模板对求解效率的影响，UNK数量变化。
    #    (3)
    # 1. compare other invariant generation engine [no disjunctions and strengthen]
    #   讨论模板方法的优势和劣势，相较于其他方法。
    # 2. explore the influence with disjunction and strengthen [about efficiency and ..]
