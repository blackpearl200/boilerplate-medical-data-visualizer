import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
# Calculate BMI and determine if the person is overweight
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)) > 25
df['overweight'] = df['overweight'].astype(int)  # Convert boolean to int (0 or 1)

# 3
# Normalize cholesterol and gluc
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4
def draw_cat_plot():
    # 5
    # Create a DataFrame for the cat plot
    df_cat = pd.melt(df, id_vars='cardio', value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6
    # Group and reformat the data
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7
    # Draw the categorical plot
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar')

    # 8
    fig.savefig('catplot.png')
    return fig

# 10
def draw_heat_map():
    # 11
    # Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                  (df['height'] >= df['height'].quantile(0.025)) &
                  (df['height'] <= df['height'].quantile(0.975)) &
                  (df['weight'] >= df['weight'].quantile(0.025)) &
                  (df['weight'] <= df['weight'].quantile(0.975))]

    # 12
    # Calculate the correlation matrix
    corr = df_heat.corr()

    # 13
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # 15
    # Plot the heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8}, ax=ax)

    # 16
    fig.savefig('heatmap.png')
    return fig

# Uncomment the following lines to draw the plots
draw_cat_plot()
draw_heat_map()
