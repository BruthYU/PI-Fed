import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


palette_dict = {"Training Time":sns.color_palette("tab10")[1],"Data Size":sns.color_palette("tab10")[0]}


plt.figure(figsize=(9,6), dpi=400)

# orders = ["2","5","10","15","20"]
orders = ['FedAvg', 'PI-Fed', 'FedNova', 'SCAFFOLD']
# para_time = {"2":2.88,"5":3.65,"10":5.84,"15":8.14,"20":10.47}
# serial_time = {"2":5.32,"5":12.53,"10":25.35,"15":36.79,"20":47.85}
time= {'FedAvg': 1.0, 'PI-Fed': 1.03 , 'FedNova': 1.14, 'SCAFFOLD': 1.34}
datasize= {'FedAvg':1.0, 'PI-Fed':1.09 , 'FedNova': 1.0, 'SCAFFOLD': 2.0}

sns.set(style="whitegrid")

data = {"Metric":[],"Value":[],"Method":[]}

for x in orders:
    data["Metric"].append("Training Time")
    data["Value"].append(time[x])
    data["Method"].append(x)
    data["Metric"].append("Data Size")
    data["Value"].append(datasize[x])
    data["Method"].append(x)

dt = pd.DataFrame(data)



ax = sns.barplot(data=dt, x="Method", y="Value", hue="Metric", width=0.50,errwidth=0.1,
                 alpha=1,edgecolor="black")
for i in ax.containers:
    ax.bar_label(i,size=16)

# ax.set(xlabel="Population Size",ylabel="Average Time per Generation (s)")
# ax.set_xlabel(xlabel="Population Size",fontsize=20)
# ax.set_ylabel(ylabel="Average Time per Generation (s)",fontsize=18)
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_xticklabels(orders, size = 20,fontdict={'weight': 'bold'})
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

ax.set_yticklabels(["{:.2f}".format(i) for i in ax.get_yticks()], fontsize = 20)
ax.legend().set_title('')
ax.legend(fontsize=20)
# plt.title("Caltech101")
# ax.axes.set_title("Caltech101",fontsize=20)
plt.tight_layout()
plt.savefig(f'./figure/efficiency.jpg')
plt.show()
pass