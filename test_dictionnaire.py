import pandas as pd

# Load the data
df = pd.read_csv('twitch_game_data.csv')

# Group by 'game' and filter
games_in_multiple_years = df.groupby('game').filter(lambda x: x['year'].nunique() > 1)
print(games_in_multiple_years)