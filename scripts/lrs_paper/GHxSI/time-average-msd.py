
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---- Adobe / Illustrator friendly PDFs ----
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

PALETTE = {
    "GH": 'steelblue',     
    "SI": 'darkorange',}

HUE_ORDER = ["GH", "SI"]


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/time_average_msd.csv')
df1['condition'] = 'GH'
df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/time_average_msd.csv')
df2['condition'] = 'SI'

df = pd.concat([df1, df2], ignore_index=True)



for cond in df['condition'].unique():
    subset = df[df['condition'] == cond]

    plt.figure(figsize=(1.5, 1.5))
    ax = sns.lineplot(data=subset, x='tau', y='msd', errorbar=('ci', 95), color=PALETTE[cond], label=cond)

    plt.xlabel('Tau', fontsize=12,fontweight='bold')
    plt.ylabel('MSD', fontsize=12,fontweight='bold')

    sns.despine()
    ax.legend(frameon=False, title=None, fontsize=11, loc="upper left")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(f'/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/GHxSI/time_average_msd_{cond}.pdf', format='pdf', bbox_inches='tight')
    plt.close()
