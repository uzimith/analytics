import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('filename', action='store', type=str, help='')
args = parser.parse_args()

data = pd.read_csv(args.filename)
sns.pointplot(x="average", y="accuracy", hue="ISI", data=data,
              markers=["^", "o"], linestyles=["-", "--"]);

font = {'family' : 'Times New Roman', 'weight' : 'bold', 'size'   : 10}
plt.rc('font', **font)
plt.xlabel("Average")
plt.ylabel("Accuracy")
plt.savefig('log/shuron/accuracy_average.eps', dpi=300)
