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

import pandas as pd

rows = []

# Existing control lines
for control in range(1, 4):  # control_1 to control_3
    for rep in range(1, 5):
        rows.append({
            "sample_id": f"control_{control}_{rep}",
            "condition": f"control_{control}",
            "replicate": rep
        })

# Trip lines (1–22)
for trip in range(1, 23):  # trip_1 to trip_22
    for rep in range(1, 5):
        rows.append({
            "sample_id": f"trip_{trip}_{rep}",
            "condition": f"trip_{trip}",
            "replicate": rep
        })

df = pd.DataFrame(rows)

# Save to CSV
df.to_csv("/Users/cochral/Desktop/sample_structure.csv", index=False)

print(df.head())
print(df.tail())

