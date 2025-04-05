import csv
import matplotlib as mpl
mpl.use('Agg')  # noqa
import matplotlib.pyplot as plt
import numpy as np
import argparse
import fnmatch
import os

from collections import defaultdict



from benchmark.datasets import DATASETS
from benchmark.algorithms.definitions import get_definitions
from benchmark.plotting.metrics import all_metrics as metrics
from benchmark.plotting.utils import (get_plot_label, compute_metrics,
        create_linestyles, create_pointset, find_configuration_with_fixed_recall,  compute_cdf_to_robustness_values, find_configuration_with_fixed_range)
from benchmark.results import (store_results, load_all_results,
                            get_unique_algorithms)

def create_plot_with_fixed_metric(all_data, lims, raw, x_scale, y_scale, xn, yn, fn_out, linestyles, fix_param, fix_min, fix_max):
    fixm = metrics[fix_param]
    ym = metrics[yn]
    selected_algorithms = {}
    selected_data = {}
    print(all_data.keys())
    print(lims.keys())
    for algo in all_data.keys():
        if(selected_algorithms.get(algo) == None):
            selected_algorithms[algo] = []
        selected_algorithms[algo] = find_configuration_with_fixed_range(lims[algo], fixm, ym, fix_min, fix_max)
        selected_data[algo] = []
        for data in all_data[algo]:
            for selected in selected_algorithms[algo]:
                if data[1] == selected[1]:
                    selected_data[algo].append(data)
                    break
    description = ""
    if fix_min != 0 and fix_max != float('inf'):
        description = "%s <= "%(fix_min) + fix_param + " <= %s"%(fix_max)
    elif fix_min != 0:
        description = fix_param + " >= %s"%(fix_min)
    elif fix_max != float('inf'):
        description = fix_param + " <= %s"%(fix_max)

    create_plot(selected_data, raw, x_scale, y_scale, xn, yn, fn_out, linestyles, description)


def create_plot_cdf(all_data_original, all_cdfs, raw, xn, yn, fn_out, linestyles, fix_recall=0.95):
    print("Creating Robustness Plot")
    xm, ym = (metrics[xn], metrics[yn])
    selected_algorithms = {}
    selected_cdf_data = []
    
    
    for algo in all_data_original.keys():
        selected_algorithms[algo] = find_configuration_with_fixed_recall(all_data_original[algo], xm, ym, fix_recall)
        print("Selected configuration for %s" % (selected_algorithms[algo][0]), end = " ")
        print(selected_algorithms[algo])
        for cdf_data in all_cdfs[algo]:
            if(cdf_data[1] == selected_algorithms[algo][1]):
                selected_cdf_data.append(cdf_data)
                break
    print("Selected robustness data")
    for cdf_data in selected_cdf_data:
        print(np.array(cdf_data[3]))

    labels = []
    plt.figure(figsize=(12, 9))
    config = []
    for cdf_data in selected_cdf_data:
        xs = np.array(cdf_data[-2])
        ys = np.array(cdf_data[-1])
        algo = cdf_data[0]
        color, faded, linestyle, marker = linestyles[algo]
        plt.plot(
            xs, ys, "-", label=algo, color=color, ms=7, mew=3, lw=3, marker=marker
        )
        print(xs, ys)
        labels.append(algo)
        config.append(cdf_data[1])
    #plt.plot(np.arange(0, 1.1, 0.1), 
    #         np.array([1, 0.97218, 0.96806, 0.96484, 0.9608,  0.95476, 0.94445, 0.92665, 0.88612, 0.81303, 0.64913]),
    #         "-", label="SPANN", color="black", ms=7, mew=3, lw=3, marker="o")
    #labels.append("SPANN")
    ax = plt.gca()
    ax.set_xlabel("Recall Rate (Threshold)")
    ax.set_ylabel("Robustness Value (Frequency of Recall Rate >= x%)") 
    ax.set_title("Robustness Plot with Different Threshold Values, Fixed Configuration with Recall Rate = %.2f" % (fix_recall))
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 9})
    # plt.ylim(0, 1)
    plt.grid(visible=True, which="major", color="0.65", linestyle="-")
    plt.xticks(np.arange(0, 1.1, 0.1))
    # for idx, conf in enumerate(config):
        # conf is the str representation of the configuration
        # we add them at the bottom of the plot
        # plt.text(0.5, 0.01 - idx * 0.01, conf, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    plt.setp(ax.get_xminorticklabels(), visible=True)
    plt.savefig(fn_out, bbox_inches="tight")
    plt.close()

def create_plot(all_data, raw, x_scale, y_scale, xn, yn, fn_out, linestyles, description=""):
    xm, ym = (metrics[xn], metrics[yn])
    # Now generate each plot
    handles = []
    labels = []
    plt.figure(figsize=(12, 9))

    # Sorting by mean y-value helps aligning plots with labels
    def mean_y(algo):
        xs, ys, ls, axs, ays, als = create_pointset(all_data[algo], xn, yn)
        return -np.log(np.array(ys)).mean()
    # Find range for logit x-scale
    min_x, max_x = 1, 0
    min_y, max_y = 1, 0
    # print the title of the figure to a log file, adding not to overwrite
    # log_file = open("export_with_label-ds-%s_ivf.log" % (args.dataset), "a")
    # log_file.write(linestyles)
    # log_file.write("# " + get_plot_label(xm, ym) + "\n")
    for algo in sorted(all_data.keys(), key=mean_y):
        xs, ys, ls, axs, ays, als = create_pointset(all_data[algo], xn, yn)
        # log_file.write(algo + "_" + xn + "_" + yn + " = " + str(axs) + "\n")
        # log_file.write(algo + "_" + yn + "_" + xn + " = " + str(ays) + "\n")
        # log_file.write(algo + "_label = " + str(als) + "\n")
        min_x = min([min_x]+[x for x in axs if x > 0])
        max_x = max([max_x]+[x for x in axs if x < 1])
        min_y = min([min_y]+[y for y in ays if y > 0])
        max_y = max([max_y]+[y for y in ays])
        color, faded, linestyle, marker = linestyles[algo]
        handle, = plt.plot(xs, ys, '-', label=algo, color=color,
                           ms=7, mew=3, lw=3, linestyle=linestyle,
                           marker=marker)
        handles.append(handle)
        if raw:
            # print(ays)
            handle2, = plt.plot(axs, ays, 'o', label=algo, color=color,
                                ms=5, mew=2, lw=2, linestyle=linestyle,
                                marker=marker)
        labels.append(algo)
    # log_file.close()
    ax = plt.gca()
    ax.set_ylabel(ym['description'])
    ax.set_xlabel(xm['description'])
    # Custom scales of the type --x-scale a3
    if x_scale[0] == 'a':
        if x_scale[1:] == 'neurips23ood':
          alpha = 3
        else:
          alpha = int(x_scale[1:])
        fun = lambda x: 1-(1-x)**(1/alpha)
        inv_fun = lambda x: 1-(1-x)**alpha
        ax.set_xscale('function', functions=(fun, inv_fun))
        if x_scale[1:] == 'neurips23ood':
          xm['lim'] = (0.7, 0.97)
          plt.xticks([0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
        elif alpha <= 3:
            ticks = [inv_fun(x) for x in np.arange(0,1.2,.2)]
            plt.xticks(ticks)
        elif alpha > 3:
            from matplotlib import ticker
            ax.xaxis.set_major_formatter(ticker.LogitFormatter())
            #plt.xticks(ticker.LogitLocator().tick_values(min_x, max_x))
            plt.xticks([0, 1/2, 1-1e-1, 1-1e-2, 1-1e-3, 1-1e-4, 1])
    # Other x-scales
    else:
        ax.set_xscale(x_scale)

    if y_scale[0] == "a":
        alpha = float(y_scale[1:])

        def fun(y):
            return 1 - (1 - y) ** (1 / alpha)

        def inv_fun(y):
            return 1 - (1 - y) ** alpha

        ax.set_yscale("function", functions=(fun, inv_fun))
        if alpha <= 3:
            ticks = [inv_fun(y) for y in np.arange(0, 1.2, 0.2)]
            plt.yticks(ticks)
        if alpha > 3:
            from matplotlib import ticker

            ax.yaxis.set_major_formatter(ticker.LogitFormatter())
            # plt.yticks(ticker.LogitLocator().tick_values(min_x, max_x))
            # plt.yticks([0, 1 / 2, 1 - 1e-1, 1 - 1e-2, 1 - 1e-3, 1 - 1e-4, 1])
            plt.yticks([0, 1 / 2, 1 - 1e-1, 1-(1e-1)/2, 1 - 1e-2, 1 - 1e-3, 1 - 1e-4, 1])
    # Other y-scales
    else:
        ax.set_yscale(y_scale)
    # ax.set_yscale(y_scale)
    ax.set_title(get_plot_label(xm, ym) + ", " + description)
    box = plt.gca().get_position()
    # plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles, labels, loc='center left',
              bbox_to_anchor=(1, 0.5), prop={'size': 9})
    plt.grid(visible=True, which='major', color='0.65', linestyle='-')
    plt.setp(ax.get_xminorticklabels(), visible=True)

    # Logit scale has to be a subset of (0,1)
    if 'lim' in xm and x_scale != 'logit':
        x0, x1 = xm['lim']
        plt.xlim(max(x0,0), min(x1,1))
    elif x_scale == 'logit':
        plt.xlim(min_x, max_x)
    if 'lim' in ym:
        plt.ylim(ym['lim'])

    if x_scale == 'a4':
        plt.xlim(min(1/2, min_x), max(max_x, 1))
    if x_scale == 'linear' and y_scale == 'linear':
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
    # Workaround for bug https://github.com/matplotlib/matplotlib/issues/6789
    ax.spines['bottom']._adjust_location()

    plt.savefig(fn_out, bbox_inches='tight')
    plt.close()

class allowed_metrics(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        self.valid_patterns = metrics.keys()
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if not any(fnmatch.fnmatch(values, pattern) for pattern in self.valid_patterns):
            raise argparse.ArgumentError(self, f"Invalid choice: {values} (choose from {', '.join(self.valid_patterns)})")
        setattr(namespace, self.dest, values)


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        metavar="DATASET",
        required=True)
    parser.add_argument(
        '--count',
        default=-1,
        type=int)
    parser.add_argument(
        '--csv',
        metavar='FILE',
        help='use results from pre-computed CSV file',
    )
    parser.add_argument(
        '--definitions',
        metavar='FILE',
        help='load algorithm definitions from FILE',
        default='algos-2021.yaml')
    parser.add_argument(
        '--limit',
        default=-1)
    parser.add_argument(
        '-o', '--output')
    parser.add_argument(
        '-x', '--x-axis',
        help='Which metric to use on the X-axis',
        action=allowed_metrics,
        default="k-nn")
    parser.add_argument(
        '-y', '--y-axis',
        help='Which metric to use on the Y-axis',
        action=allowed_metrics,
        default="qps")
    parser.add_argument(
        '-X', '--x-scale',
        help='Scale to use when drawing the X-axis. Typically linear, logit or a2',
        default='linear')
    parser.add_argument(
        '-Y', '--y-scale',
        help='Scale to use when drawing the Y-axis',
        choices=["linear", "log", "symlog", "logit"],
        default='linear')
    parser.add_argument(
        '--raw',
        help='Show raw results (not just Pareto frontier) in faded colours',
        action='store_true')
    parser.add_argument(
        '--recompute',
        help='Clears the cache and recomputes the metrics',
        action='store_true')
    parser.add_argument(
        '--neurips23track',
        choices=['filter', 'ood', 'sparse', 'streaming', 'none'],
        default='none'
    )
    parser.add_argument(
        '--private-query',
        help='Use the private queries and ground truth',
        action='store_true')
    

    parser.add_argument(
        "-T", 
        "--plot-type", 
        choices=["normal", "cdf", "filter"],
        default="normal", 
        help="The type of plot to generate, for simple 2 metrics use normal, for robustness figure use cdf, if filtering one metric use filter")
    parser.add_argument("--fix-recall", help="Fix the recall value for robustness figure", default=90)
    parser.add_argument(
        "--fix-metric", 
        help="Fix the metric value",  
        default="",
        action=allowed_metrics,
    )
     
    parser.add_argument("--min", help="Minimum value for fixed metric", default=0)
    parser.add_argument("--max", help="Maximum value for fixed metric", default=float('inf'))
    args = parser.parse_args()

    if not args.output:
        path = "results/figures/%s/%d" % (args.dataset, args.count)
        os.makedirs("results/figures/%s/%d" % (args.dataset, args.count), exist_ok=True)
        args.output = path + "/%s.png" % \
        (args.dataset
         + "-%s-%s-%s-%s" % (args.x_axis, args.x_scale, args.y_axis, args.y_scale) 
         + ("-robustness%s"%args.fix_recall if args.plot_type == "cdf" else "") 
         + ("-fixed_%s"%args.fix_metric if args.plot_type == "filter" else "") 
         + ("-min%s-max%s"%(str(args.min), str(args.max)) if args.plot_type == "filter" else "") 
         + ("-raw" if args.raw else ""))
        print("writing output to %s" % args.output)

    dataset = DATASETS[args.dataset]()

    if args.count == -1:
        args.count = dataset.default_count()

    if args.x_axis == "k-nn" and dataset.search_type() == "range":
        args.x_axis = "ap"


    count = int(args.count)
    if not args.csv:
        unique_algorithms = get_unique_algorithms()
        results = load_all_results(args.dataset, count, neurips23track=args.neurips23track)
        if args.private_query:
            runs = compute_metrics(dataset.get_private_groundtruth(k=args.count),
                                    results, args.x_axis, args.y_axis, args.recompute)
        else:
            runs = compute_metrics(dataset.get_groundtruth(k=args.count),
                                    results, args.x_axis, args.y_axis, args.recompute)
    else:
        with open(args.csv) as csvfile:
            reader = csv.DictReader(csvfile)
            data = [row for row in reader if row['dataset'] == args.dataset and
                        row['track'] == args.neurips23track]
            runs = defaultdict(list)
            for result in data:
                # we store a single quality metric in the csv file
                x_axis = args.x_axis
                if x_axis == 'k-nn' or x_axis == 'ap':
                    x_axis='recall/ap'
                y_axis = args.y_axis
                if y_axis == 'k-nn' or y_axis == 'ap':
                    y_axis='recall/ap'
                runs[result['algorithm']].append((result['algorithm'], result['parameters'],
                                                  float(result[x_axis]), float(result[y_axis])))
            unique_algorithms = set(runs)


    linestyles = create_linestyles(sorted(unique_algorithms))


    if not runs:
        raise Exception('Nothing to plot')
    
    if args.plot_type == "cdf":
        results = load_all_results(args.dataset, count, neurips23track=args.neurips23track)
        robustness_values = compute_cdf_to_robustness_values(dataset.get_groundtruth(k=args.count), results, args.x_axis, args.y_axis, args.recompute)
        print("Robustness values")
        print(robustness_values)
        create_plot_cdf(runs, robustness_values, args.raw, args.x_axis, args.y_axis, args.output, linestyles, float(args.fix_recall)/100.0)
    elif args.plot_type == "filter":
        results = load_all_results(args.dataset, count, neurips23track=args.neurips23track)
        lims = compute_metrics(dataset.get_groundtruth(k=args.count), results, args.fix_metric, args.y_axis, args.recompute)
        create_plot_with_fixed_metric(
            runs, lims, args.raw, args.x_scale, args.y_scale, args.x_axis, args.y_axis, args.output, linestyles, args.fix_metric, args.min, args.max
        )
    else:
        create_plot(
            runs, args.raw, args.x_scale, args.y_scale, args.x_axis, args.y_axis, args.output, linestyles
        )
