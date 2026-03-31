import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pyarrow.feather as feather
import cv2
import numpy as np
import matplotlib as mpl


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

df = pd.read_csv('/Users/cochral/Downloads/iso_1596.csv')

print(df.columns)

plt.figure(figsize=(5, 5))

# Track 1
# plt.scatter(df['Track_1 x_tail'], df['Track_1 y_tail'],
#             c=df['Normalized Frame'], s=8, cmap='Blues', label='T1 tail')
plt.scatter(df['Track_1 x_body'], df['Track_1 y_body'],
            c=df['Normalized Frame'], s=8, cmap='Blues', label='T1 body')
# plt.scatter(df['Track_1 x_head'], df['Track_1 y_head'],
#             c=df['Normalized Frame'], s=8, cmap='Blues', label='T1 head')

# Track 2
# plt.scatter(df['Track_2 x_tail'], df['Track_2 y_tail'],
#             c=df['Normalized Frame'], s=8, cmap='Reds', label='T2 tail')
plt.scatter(df['Track_2 x_body'], df['Track_2 y_body'],
            c=df['Normalized Frame'], s=8, cmap='Reds', label='T2 body')
# plt.scatter(df['Track_2 x_head'], df['Track_2 y_head'],
#             c=df['Normalized Frame'], s=8, cmap='Reds', label='T2 head')

plt.gca().set_aspect('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Normalized Frame')
plt.legend(markerscale=2, fontsize=8)
plt.savefig('/Users/cochral/Downloads/iso_1596_trajectory.pdf', format='pdf', bbox_inches='tight')
plt.close()



plt.figure(figsize=(4, 4))

sns.lineplot(data=df, x='Normalized Frame', y='track1_angle', label='Track 1', color='darkblue')
sns.lineplot(data=df, x='Normalized Frame', y='track2_angle', label='Track 2', color='darkred')

plt.ylabel('Angle', fontsize=12, fontweight='bold')
plt.xlabel('Normalized Frame', fontsize=12, fontweight='bold')
plt.axvline(0, color='gray', linestyle='--')
plt.savefig('/Users/cochral/Downloads/iso_1596_angle.pdf', format='pdf', bbox_inches='tight')
plt.show()


plt.figure(figsize=(4, 4))
sns.lineplot(data=df, x='Normalized Frame', y='min_distance', label='Track 1', color='gray')

plt.ylabel('Distance', fontsize=12, fontweight='bold')
plt.xlabel('Normalized Frame', fontsize=12, fontweight='bold')
plt.axvline(0, color='gray', linestyle='--')
plt.savefig('/Users/cochral/Downloads/iso_1596_distance.pdf', format='pdf', bbox_inches='tight')
plt.show()




plt.figure(figsize=(4, 4))

sns.lineplot(data=df, x='Normalized Frame', y='track1_speed', label='Track 1', color='darkblue')
sns.lineplot(data=df, x='Normalized Frame', y='track2_speed', label='Track 2', color='darkred')

plt.ylabel('Speed', fontsize=12, fontweight='bold')
plt.xlabel('Normalized Frame', fontsize=12, fontweight='bold')
plt.axvline(0, color='gray', linestyle='--')
plt.savefig('/Users/cochral/Downloads/iso_1596_speed.pdf', format='pdf', bbox_inches='tight')
plt.show()


plt.figure(figsize=(4, 4))

sns.lineplot(data=df, x='Normalized Frame', y='track1_acceleration', label='Track 1', color='darkblue')
sns.lineplot(data=df, x='Normalized Frame', y='track2_acceleration', label='Track 2', color='darkred')

plt.ylabel('Acceleration', fontsize=12, fontweight='bold')
plt.xlabel('Normalized Frame', fontsize=12, fontweight='bold')
plt.axvline(0, color='gray', linestyle='--')
plt.savefig('/Users/cochral/Downloads/iso_1596_acceleration.pdf', format='pdf', bbox_inches='tight')
plt.show()