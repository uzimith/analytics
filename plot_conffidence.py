import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('filename', action='store', type=str, help='')
args = parser.parse_args()

data = pd.read_csv(args.filename)
sns.stripplot(x='answer', y='confidence', split=True, hue='name', data=data)
plt.show()
