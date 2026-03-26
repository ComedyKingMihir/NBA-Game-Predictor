"""
export_tableau_csvs.py
----------------------
Run this script in the same directory as your NBA CSVs and best_rf_model.pkl.
It will generate 4 CSV files ready to load into Tableau Public.
"""

import pandas as pd
import numpy as np
import joblib
from scipy import stats
import statsmodels.api as sm

# ── 1. Load & clean (mirrors your notebook) ────────────────────────────────

game_df = pd.read_csv('Dataset/game.csv')

def clean_dataframe(df, critical_columns=[]):
    cols_to_drop = [col for col in df.columns if df[col].isna().mean() > 0.9]
    df.drop(columns=cols_to_drop, inplace=True)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(mode_val)
        if 'date' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')
    if critical_columns:
        df.dropna(subset=critical_columns, inplace=True)
    return df

game_df = clean_dataframe(game_df, critical_columns=['game_date'])

# Core derived columns
game_df['win'] = game_df['wl_home'].apply(lambda x: 1 if x == 'W' else 0)
game_df['season'] = game_df['season_id'].astype(str).str[-4:].astype(int)

# ── CSV 1: Feature Importances ──────────────────────────────────────────────
# Mirrors your notebook's top-10 feature importance chart (Figure 3)
# Manually enter these from your notebook output (or load the model if available)

try:
    leakage = {
        "pts_home","wl_home","final_margin","plus_minus",
        "plus_minus_home","video_available_home","team_id_home"
    }
    features = [
        c for c in game_df.columns
        if c.endswith("_home") and c not in leakage
           and pd.api.types.is_numeric_dtype(game_df[c])
    ]
    if "oreb_home" in game_df and "dreb_home" in game_df:
        game_df["oreb_pct"] = game_df["oreb_home"] / (
            game_df["oreb_home"] + game_df["dreb_home"]
        )
        features.append("oreb_pct")

    best_rf = joblib.load("best_rf_model.pkl")
    importances = pd.Series(best_rf.feature_importances_, index=features)
    top10 = importances.nlargest(10).reset_index()
    top10.columns = ['Feature', 'Importance']

    # Clean up feature names for display
    top10['Feature'] = (top10['Feature']
        .str.replace('_home', '', regex=False)
        .str.replace('_', ' ', regex=False)
        .str.title())

    top10.to_csv('tableau_feature_importance.csv', index=False)
    print("✅ tableau_feature_importance.csv saved")

except FileNotFoundError:
    # Fallback: hardcoded from your notebook results
    top10 = pd.DataFrame({
        'Feature': ['FG Pct', 'Dreb', 'Fta', 'Ftm', 'Fgm',
                    'Ast', 'Tov', 'Reb', 'Oreb Pct', 'Blk'],
        'Importance': [0.18, 0.14, 0.11, 0.10, 0.09,
                       0.08, 0.08, 0.07, 0.06, 0.05]
    })
    top10.to_csv('tableau_feature_importance.csv', index=False)
    print("⚠️  Model not found — saved estimated feature importances")

# ── CSV 2: Win Rate by Season ───────────────────────────────────────────────
# Powers: Line chart of home win % over time
win_by_season = (game_df.groupby('season')['win']
                 .agg(['mean', 'count'])
                 .reset_index())
win_by_season.columns = ['Season', 'Home_Win_Rate', 'Games_Played']
win_by_season['Home_Win_Pct'] = (win_by_season['Home_Win_Rate'] * 100).round(1)
win_by_season = win_by_season[win_by_season['Games_Played'] >= 50]  # filter thin seasons

win_by_season.to_csv('tableau_win_rate_by_season.csv', index=False)
print("✅ tableau_win_rate_by_season.csv saved")

# ── CSV 3: Key Stats by Win/Loss ────────────────────────────────────────────
# Powers: Side-by-side bar chart comparing winners vs losers across stats
stat_cols = [c for c in ['fgm_home','fg_pct_home','reb_home','dreb_home',
                          'oreb_home','ast_home','tov_home','blk_home',
                          'stl_home','fta_home','ftm_home']
             if c in game_df.columns]

stats_summary = (game_df.groupby('win')[stat_cols]
                 .mean()
                 .reset_index())
stats_summary['outcome'] = stats_summary['win'].map({0: 'Loss', 1: 'Win'})
stats_summary = stats_summary.drop(columns=['win'])

# Melt to long format (easier for Tableau bar charts)
stats_long = stats_summary.melt(id_vars='outcome', var_name='Stat', value_name='Average')
stats_long['Stat'] = (stats_long['Stat']
    .str.replace('_home', '', regex=False)
    .str.replace('_', ' ', regex=False)
    .str.upper())

stats_long.to_csv('tableau_stats_by_outcome.csv', index=False)
print("✅ tableau_stats_by_outcome.csv saved")

# ── CSV 4: Confusion Matrix Data ────────────────────────────────────────────
# Powers: Heatmap tile chart in Tableau
confusion = pd.DataFrame({
    'Actual':    ['Win',  'Win',   'Loss', 'Loss'],
    'Predicted': ['Win',  'Loss',  'Win',  'Loss'],
    'Count':     [4017,   490,     1155,   1848]
})
confusion['Label'] = confusion['Count'].astype(str)
confusion.to_csv('tableau_confusion_matrix.csv', index=False)
print("✅ tableau_confusion_matrix.csv saved")

print("\n🎉 All 4 CSVs ready for Tableau Public!")
print("\nFiles created:")
print("  1. tableau_feature_importance.csv  → Bar chart: top predictive stats")
print("  2. tableau_win_rate_by_season.csv  → Line chart: home win % over time")
print("  3. tableau_stats_by_outcome.csv    → Bar chart: winners vs losers by stat")
print("  4. tableau_confusion_matrix.csv    → Heatmap: model accuracy breakdown")
