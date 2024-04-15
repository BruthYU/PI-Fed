import argparse
import pickle
import os
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='Last Acc or Avg Acc')
parser.add_argument('--plot_type', required=True, type=str)
args = parser.parse_args()
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

colors = ['blue', 'red', 'orange','green']
method_names = ['FedAvg', 'FedNova', 'PI_Fed', 'SCAFFOLD']
method_dicts = {}
for name in method_names:
    log_file_path = os.path.join("../res/cifar100_10/", name, "alg_info.pkl")
    with open(log_file_path, 'rb') as f:
        log = pickle.load(f)
    method_dicts[name] = log


t = list(range(1, 11))


fig, axs = plt.subplots(2, 4, figsize=(8,2),dpi=800)
for i in range(4):
    axs[i].set_box_aspect(0.85)

attr_name = ''
if args.plot_type == 'latest':
    attr_name = 'latest_acc'
elif args.plot_type == 'average':
    attr_name = 'average_acc'
else:
    raise ValueError(f"No plot type named {args.plot_type}")

for name in method_names:
    ys = getattr(method_dicts[name], attr_name)
    axs[0].plot(t, ys, label = f'name',linewidth=1.8,color=colors[0])
axs[0].set_xlim(0,11)
axs[0].set_ylim(0.58,0.80)
axs[0].xaxis.set_major_locator(MultipleLocator(1))
axs[0].yaxis.set_major_locator(MultipleLocator(0.1))
axs[0].set_xlabel('Edits',fontweight='bold',family = 'serif')
axs[0].set_ylabel('Edit Success',fontweight='bold',family = 'serif')
axs[0].grid(True)

