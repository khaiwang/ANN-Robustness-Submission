import argparse
import os

def generate_plot_commands(dataset=None, count=10, min_recall=0.70, max_recall=0.95):
    robustness = [0.1, 0.3, 0.5, 0.7, 0.9]
    fixed_recall = 90
    datasets = []
    if dataset is None:
        datasets = ["msspacev-10M", "deep-10M", "text2image-10M", "msmarco-10M"]
    else:
        datasets.append(dataset)
    commands = []
    for dataset in datasets:
        commands.append(f"python plot.py -x k-nn -y qps --dataset {dataset} --neurips23track ood --count {count} --recompute --fix-metric k-nn --min {min_recall} --max {max_recall} -T filter")
        for r in robustness:
            commands.append(f"python plot.py -x robustness@{r} -y qps --dataset {dataset} --neurips23track ood --count {count} --fix-metric k-nn --min {min_recall} --max {max_recall} -T filter")
            commands.append(f"python plot.py -x k-nn -y robustness@{r} --dataset {dataset} --neurips23track ood --count {count} --fix-metric k-nn --min {min_recall} --max {max_recall} -T filter")
        commands.append(f"python plot.py -x k-nn -y qps --raw --dataset {dataset} --neurips23track ood --count {count} -T cdf --fix-recall {fixed_recall}")
        
    return commands

def generate_install_commands():
    indices = ["scann", "zilliz", "diskann", "faiss", "faiss_hnsw", "puck"]
    print("Installing the index dockers.")
    print("Dockers to install:")
    print(indices)
    commands = []
    for index in indices:
        commands.append(f"python install.py --neurips23track ood --algorithm {index}")
    return commands

def generate_run_commands(dataset=None, count=10):
    indices = ["scann", "zilliz", "diskann", "faiss-ivf", "faiss-ivfpqfs", "faiss_hnsw", "puck"]
    datasets = []
    if dataset is None:
        datasets = ["msspacev-10M", "deep-10M", "text2image-10M", "msmarco-10M"]
    else:
        datasets.append(dataset)
    print("Running the index dockers.")
    print("Dockers to run:")
    print(indices)
    print("Datasets to run:")
    print(datasets)
    commands = []
    for index in indices:
        for dataset in datasets:
            commands.append(f"python run.py --neurips23track ood --algorithm {index} --dataset {dataset} --count {count}")
    return commands

def generate_run_k100_commands():
    """Run HNSW with K=100 on text2image for the retrieve-and-rerank analysis (§5.4)."""
    commands = [
        "python run.py --neurips23track ood --algorithm faiss_hnsw --dataset text2image-10M --count 100 --force"
    ]
    return commands


def generate_analysis_commands():
    """Run metric comparison (§4) and index family analysis (§5.4)."""
    commands = [
        'bash run_metric_comparison.sh',
        'PYTHONPATH="." python analyze_index_families.py',
    ]
    return commands


def main():
    parser = argparse.ArgumentParser(description="Generate plot commands with a specified dataset.")
    parser.add_argument("--dataset", type=str, required=False, help="Specify the dataset to use.")
    parser.add_argument("--run", type=str, required=True,
                        choices=["plot", "install", "run", "run-k100", "analyze"],
                        help="Specify the mode: plot, install, run (§5.1), run-k100 (§5.4), analyze (§4+§5.4).")
    parser.add_argument("--count", type=int, required=False, help="Specify the number of topk.", default=10)
    parser.add_argument("--max_recall", type=int, required=False, help="Specify the max recall.", default=0.95)
    parser.add_argument("--min_recall", type=int, required=False, help="Specify the min recall.", default=0.70)
    args = parser.parse_args()
    if args.run == "plot":
        commands = generate_plot_commands(args.dataset, args.count, args.min_recall, args.max_recall)
    elif args.run == "install":
        commands = generate_install_commands()
    elif args.run == "run":
        commands = generate_run_commands(args.dataset, args.count)
    elif args.run == "run-k100":
        commands = generate_run_k100_commands()
    elif args.run == "analyze":
        commands = generate_analysis_commands()
    else:
        raise ValueError(f"Invalid run mode: {args.run}")
    for command in commands:
        print(command)
        os.system(command)

if __name__ == "__main__":
    main()
