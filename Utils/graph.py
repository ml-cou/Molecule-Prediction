import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

# Load the data
df = pd.read_excel('../temperature.xlsx')

# Get column names
column_names = df.columns.tolist()
y = column_names[0]
x_vars = column_names[1:]

# Create subplots using Seaborn
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Use Seaborn to plot the data
for x in x_vars:
    tmp = df.sort_values(by=x)

    # Plot a line plot using Seaborn
    sns.lineplot(data=tmp, x=y, y=x, ax=axes[0], label=x, linestyle='--', marker='o')

    # Plot a scatter plot using Seaborn
    sns.scatterplot(data=tmp, x=y, y=x, ax=axes[1], label=x)

# Set labels and legends using Seaborn
axes[0].set_xlabel(y)
axes[0].set_ylabel("Temperature")
axes[0].legend()

axes[1].set_xlabel(y)
axes[1].set_ylabel("Temperature")
axes[1].legend()

# Show the grid and the plots
plt.grid()
plt.show()