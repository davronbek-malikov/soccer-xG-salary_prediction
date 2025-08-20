import pandas as pd
import numpy as np

class FeatureEngineering:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def add_goals_features(self):
        self.df['aGg'] = self.df['goals'] / self.df['games']   # goals per game
        self.df['gpm'] = self.df['goals'] / self.df['time']    # goals per minute
        return self

    def add_assists_features(self):
        self.df['apg'] = self.df['assists'] / self.df['games']   # assists per game
        self.df['apm'] = self.df['assists'] / self.df['time']    # assists per minute
        return self

    def add_shots_features(self):
        self.df['shpg'] = self.df['shots'] / self.df['games']   # shots per game
        self.df['shpm'] = self.df['shots'] / self.df['time']    # shots per minute
        return self

    def add_keypasses_features(self):
        self.df['kppg'] = self.df['key_passes'] / self.df['games']
        self.df['kppm'] = self.df['key_passes'] / self.df['time']
        return self

    def add_cards_features(self):
        self.df['ypg'] = self.df['yellow_cards'] / self.df['games']
        self.df['ypm'] = self.df['yellow_cards'] / self.df['time']
        self.df['rpg'] = self.df['red_cards'] / self.df['games']
        self.df['rpm'] = self.df['red_cards'] / self.df['time']
        return self


    def add_xg_features(self):
        self.df['xGdiff'] = self.df['goals'] - self.df['xG']
        self.df['xGg'] = self.df['xG'] / self.df['games']
        return self

    def finalize(self):
        # Handle infinite or NaN values safely
        self.df = self.df.replace([np.inf, -np.inf], np.nan).fillna(0)
        return self.df
