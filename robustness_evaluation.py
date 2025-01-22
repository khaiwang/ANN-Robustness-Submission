import argparse

def generate_plot_commands(dataset=None):
    robustness = [0.1, 0.3, 0.5, 0.7, 0.9]
    fixed_recall = 90
    datasets = []
    if dataset is None:
        datasets = ["msspacev-10M", "deep-10M", "text2image-10M"]
    else:
        datasets.append(dataset)
    commands = []
    for dataset in datasets:
        commands.append(f"python plot.py -x k-nn -y qps --dataset {dataset} --neurips23track ood --count 10 --recompute",)
        for r in robustness:
            commands.append(f"python plot.py -x robustness@{r} -y qps --dataset {dataset} --neurips23track ood --count 10")
            commands.append(f"python plot.py -x k-nn -y robustness@{r} --raw --dataset {dataset} --neurips23track ood --count 10 -T cdf --fixed-recall {fixed_recall}")

        commands.append(f"python plot.py -x k-nn -y qps --raw --dataset {dataset} --neurips23track ood --count 10 -T cdf --fixed-recall {fixed_recall}")
        
    return commands

def generate_install_commands():
    indices = ["scann", "zilliz", "diskann", "faiss", "faiss_hnsw", "puck"]
    print("Installing the index dockers.")
    print("Dockers to install:")
    print(indices)
    commands = []
    for index in indices:
        commands.append(f"python install.py --neurips23track ood --algorithm {index}")

def generate_run_commands():
    indices = ["scann", "zilliz", "diskann", "faiss-ivf", "faiss_hnsw", "puck"]
    datasets = ["msspacev-10M", "deep-10M", "text2image-10M"]
    print("Running the index dockers.")
    print("Dockers to run:")
    print(indices)
    print("Datasets to run:")
    print(datasets)
    commands = []
    for index in indices:
        for dataset in datasets:
            commands.append(f"python run.py --neurips23track ood --algorithm {index} --dataset {dataset}")


def main():
    parser = argparse.ArgumentParser(description="Generate plot commands with a specified dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Specify the dataset to use.")
    parser.add_argument("--run", type=str, required=True, 
                        choices=["plot", "install", "run"],
                        help="Specify the mode, options are plot, install, run.")
    args = parser.parse_args()
    if args.run == "plot":
        commands = generate_plot_commands(args.dataset)
    elif args.run == "install":
        commands = generate_install_commands()
    elif args.run == "run":
        commands = generate_run_commands
    for command in commands:
        print(command)

if __name__ == "__main__":
    main()
