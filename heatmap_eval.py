import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data similar to the one in the image
data = {
    'coacAI': [0, 12, 34, 42, 12, 30, 2, 2],
    'droplet': [0, 8, 62, 36, 6, 50, 12, 50],
    'workerRushAI': [0, 20, 78, 80, 20, 76, 28, 34],
    'Izanagi': [0, 54, 46, 54, 54, 44, 16, 46],
    'mixedBot': [0, 60, 76, 80, 60, 78, 26, 68],
    'tiamat': [0, 84, 84, 84, 86, 96, 38, 80],
    'lightRushAI': [0, 82, 56, 80, 80, 52, 26, 66],
    'naiveMCTSAI': [0, 70, 76, 84, 70, 80, 50, 66],
    'guidedRojoA3N': [0, 96, 88, 80, 96, 98, 88, 94],
    'rojo': [0, 100, 98, 100, 100, 98, 94, 96],
    'randomBiasedAI': [0, 100, 98, 98, 100, 100, 100, 100],
    'randomAI': [20, 100, 100, 100, 100, 100, 100, 100],
    'passiveAI': [20, 100, 100, 100, 100, 100, 100, 100]
}

# Create a DataFrame
df = pd.DataFrame(data, index=['sparse_reward', 'shaped_reward', 'curriculum', 'only_coacAI', 'self-play', 'opponents', 'combat_unit', 'building'])


# Create the heatmap
plt.figure(figsize=(13, 20))  # Increase figure size to fit labels
ax = sns.heatmap(df, annot=True, fmt="d", cmap="YlGnBu", linewidths=.5, cbar=True, square=True)

# Rotate x-axis labels to show full names
plt.xticks(rotation=25, ha="right")  # Rotate x-axis labels

# Adjust layout to ensure everything fits without overlapping
plt.tight_layout()

# Add title and labels
plt.title('Win Percentages Heatmap')
plt.xlabel('AI Bots')
plt.ylabel('Trained agent to play 50 games')

# Show the plot
plt.show()