import argparse
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
parser = argparse.ArgumentParser(description='Last Acc or Avg Acc')
parser.add_argument('--plot_type', required=True, type=str)
args = parser.parse_args()
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

print(f"------Plot Type: {args.plot_type}-------")
attr_name = ''
colors = []
if args.plot_type == 'latest':
    attr_name = 'latest_acc'
    colors = ['blue', 'red', 'orange', 'green']
    task_y_lim = {'cifar100_10':(0.18, 0.95), 'cifar100_20':(0.18, 0.95),
                  'tinyimagenet_10':(0.08, 0.70), 'tinyimagenet_20':(0.08, 0.80)}
    y_label = 'Latest Acc.'
    y_locator = 0.2

elif args.plot_type == 'average':
    attr_name = 'average_acc'
    sns.set_style('darkgrid')
    colors = ['blue', 'red', 'orange', 'green']
    task_y_lim = {'cifar100_10': (0.60, 0.80), 'cifar100_20': (0.60, 0.80),
                  'tinyimagenet_10': (0.40, 0.55), 'tinyimagenet_20': (0.45, 0.75)}
    y_label = 'Average Acc.'
    y_locator = 0.05
else:
    raise ValueError(f"No plot type named {args.plot_type}")



method_names = ['PI_Fed', 'FedAvg', 'FedNova', 'SCAFFOLD']
task_names = ['cifar100_10', 'cifar100_20', 'tinyimagenet_10', 'tinyimagenet_20']


method_dicts = {}
for name in task_names:
    method_dicts[name] = {}
for t_name in task_names:
    for m_name in method_names:
        log_file_path = os.path.join("../res/", t_name, m_name, "alg_info.pkl")
        with open(log_file_path, 'rb') as f:
            log = pickle.load(f)
        method_dicts[t_name][m_name] = log





fig, axs = plt.subplots(1, 4, figsize=(8,2.3),dpi=800)
for i in range(4):
    axs[i].set_box_aspect(0.85)



plot_count = 0





alias_names = ['C-10', 'C-20','T-10', 'T-20']

marker_names = ['o', 's','D', '*']


for t_idx, t_name in enumerate(task_names):
    t = range(1, int(t_name[-2:])+1)
    for m_idx, m_name in enumerate(method_names):
        ys = getattr(method_dicts[t_name][m_name], attr_name)
        if args.plot_type == 'average':
            print(f"t_name: {t_name}, m_name: {m_name}, final avg: {ys[-1]}")
        label = m_name if m_name is not 'PI_Fed' else 'PI-Fed'
        linestyle = 'solid' if m_name is not 'FedNova' else 'dashed'
        axs[plot_count].plot(t, ys, label = label,linewidth=1.5,color=colors[m_idx],linestyle=linestyle)
    axs[plot_count].set_xlim(0,int(t_name[-2:])+1)
    axs[plot_count].set_ylim(task_y_lim[t_name][0], task_y_lim[t_name][1])
    axs[plot_count].xaxis.set_major_locator(MultipleLocator(int(t_name[-2:])/5))
    axs[plot_count].yaxis.set_major_locator(MultipleLocator(y_locator))
    axs[plot_count].set_xlabel('Tasks Learned', fontweight='bold', family = 'serif')
    axs[plot_count].set_ylabel(y_label,fontweight='bold',family = 'serif')
    axs[plot_count].set_title(f'{alias_names[t_idx]}', fontweight='bold', family='serif')
    axs[plot_count].grid(True)
    plot_count +=1

lines_labels = [axs[3].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='lower center', ncol=4)
plt.tight_layout()
plt.subplots_adjust(wspace=0.5)
plt.subplots_adjust(wspace=0.5)
plt.subplots_adjust(bottom=0.3)

plt.savefig(f'./figure/Dissimilar_{args.plot_type}.jpg')

# plt.show()
