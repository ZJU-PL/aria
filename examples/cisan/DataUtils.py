import numpy as np


def read_table(data_path: str):
    with open(data_path) as f:
        lines = f.readlines()
    data = []
    column_values = [[] for i in range(len(lines[0].split()))]
    for line in lines[1:]:
        values = line.strip().split()[1:]
        row = []
        for idx, val in enumerate(values):
            if val not in column_values[idx]:
                column_values[idx].append(val)
            row.append(column_values[idx].index(val))
        data.append(row)
    data = np.array(data)
    print("Data Shape:", data.shape)
    return data


def convert_npy_to_csv(npy_path: str, csv_path: str):
    data = np.load(npy_path)
    with open(csv_path, "w") as f:
        header = ["X" + str(i) for i in range(data.shape[1])]
        f.write("\t".join(header) + "\n")
        for row in data:
            f.write("\t".join([str(val) for val in row]) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--npy-path", "-n", type=str, default="data/insurance.npy")
    parser.add_argument("--csv-path", "-c", type=str, default="data/insurance-10k.csv")

    args = parser.parse_args()
    convert_npy_to_csv(args.npy_path, args.csv_path)
