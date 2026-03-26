import streamlit as st
import pandas as pd
import anthropic
import traceback
import io
import contextlib

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBA AI Analyst",
    page_icon="🏀",
    layout="wide"
)

# ── Load data ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Core game data
    game_df           = pd.read_csv("Dataset/game.csv")
    game_df['win']    = game_df['wl_home'].apply(lambda x: 1 if x == 'W' else 0)
    game_df['season'] = game_df['season_id'].astype(str).str[-4:].astype(int)
    game_df['game_date'] = pd.to_datetime(game_df['game_date'], errors='coerce')
    game_df['month']  = game_df['game_date'].dt.month
    game_df['day']    = game_df['game_date'].dt.day
    game_df['year']   = game_df['game_date'].dt.year

    # Player box scores — graceful fallback if unavailable
    try:
        details_raw = pd.read_csv("Dataset/games_details.csv")
        details_raw = details_raw.iloc[::2].reset_index(drop=True)
        details_raw['GAME_ID'] = details_raw['GAME_ID'].astype(str).str.strip()
        game_dates = game_df[['game_id', 'game_date', 'month', 'day', 'year', 'season']].copy()
        game_dates['game_id'] = game_dates['game_id'].astype(str).str.strip()
        player_df = details_raw.merge(
            game_dates, left_on='GAME_ID', right_on='game_id', how='left'
        )
    except FileNotFoundError:
        player_df = pd.DataFrame()

    try:
        players_df = pd.read_csv("Dataset/players.csv")
    except FileNotFoundError:
        players_df = pd.DataFrame()

    try:
        ranking_df = pd.read_csv("Dataset/ranking.csv")
    except FileNotFoundError:
        ranking_df = pd.DataFrame()

    try:
        teams_df = pd.read_csv("Dataset/teams.csv")
    except FileNotFoundError:
        teams_df = pd.DataFrame()

    # Pre-computed summaries
    feature_importance = pd.read_csv("tableau_feature_importance.csv")
    win_rate           = pd.read_csv("tableau_win_rate_by_season.csv")
    stats_by_outcome   = pd.read_csv("tableau_stats_by_outcome.csv")
    confusion          = pd.read_csv("tableau_confusion_matrix.csv")

    return (game_df, player_df, players_df, ranking_df, teams_df,
            feature_importance, win_rate, stats_by_outcome, confusion)

(game_df, player_df, players_df, ranking_df, teams_df,
 feature_importance, win_rate, stats_by_outcome, confusion) = load_data()

# ── Build schema context for Claude ─────────────────────────────────────────
def build_schema_context():
    fi   = feature_importance.to_string(index=False)
    wr   = win_rate.to_string(index=False)
    sbo  = stats_by_outcome.to_string(index=False)
    conf = confusion.to_string(index=False)

    return f"""
You are an expert NBA data analyst with access to a real NBA games dataset and a trained
Random Forest model (78.1% accuracy, 0.854 ROC AUC) that predicts home team wins.

You have access to the following pandas DataFrames:

1. `game_df` — {len(game_df):,} rows of team-level game stats (1946–2022)
   Columns: {game_df.columns.tolist()}
   Key columns: game_date, season, win (1=home win), month, day, year
   Sample: {game_df.head(2).to_string(index=False)}

2. `player_df` — {len(player_df):,} rows of individual player box scores merged with game dates
   Columns: {player_df.columns.tolist()}
   Key columns: PLAYER_NAME, PTS, REB, AST, STL, BLK, FGM, FGA, FG_PCT, MIN, game_date, month, day, year, season
   Sample: {player_df[['PLAYER_NAME','PTS','REB','AST','game_date','month','day','year']].head(2).to_string(index=False)}

3. `players_df` — {len(players_df):,} rows of player biographical info
   Columns: {players_df.columns.tolist()}

4. `ranking_df` — {len(ranking_df):,} rows of team standings over time
   Columns: {ranking_df.columns.tolist()}

5. `teams_df` — {len(teams_df):,} rows of team info
   Columns: {teams_df.columns.tolist()}

PRE-COMPUTED SUMMARIES:
Feature importances: {fi}
Home win rate by season: {wr}
Average stats by outcome: {sbo}
Model confusion matrix: {conf}

INSTRUCTIONS:
When a user asks a question, you have two options:

OPTION 1 — Answer directly from the summaries if the answer is already there.

OPTION 2 — Write a pandas query against the appropriate DataFrame.
Format it EXACTLY like this:

QUERY:
```python
result = player_df[...some pandas code...]
print(result)
```
EXPLANATION: [one sentence explaining what this query does]

Rules for queries:
- Always assign the final result to a variable called `result`
- Always end with print(result) or print(result.to_string())
- Available DataFrames: game_df, player_df, players_df, ranking_df, teams_df
- pandas (pd) is available — do not import anything else
- For date questions (e.g. "April 18"): use month and day columns in player_df or game_df
- For player questions: use player_df (has PLAYER_NAME, PTS, REB, AST etc.)
- For team standing questions: use ranking_df
- Keep queries concise and correct

If answering directly, keep it under 150 words in plain English.
"""

# ── Execute pandas query safely ──────────────────────────────────────────────
def execute_query(code: str) -> str:
    local_vars = {
        "game_df":    game_df,
        "player_df":  player_df,
        "players_df": players_df,
        "ranking_df": ranking_df,
        "teams_df":   teams_df,
        "pd":         pd
    }
    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, {"pd": pd}, local_vars)
        output = stdout_capture.getvalue()
        if not output and "result" in local_vars:
            output = str(local_vars["result"])
        return output.strip() if output else "Query ran but returned no output."
    except Exception:
        return f"Query error:\n{traceback.format_exc()}"

# ── Parse Claude's response for a query block ────────────────────────────────
def extract_query(text: str):
    if "QUERY:" not in text:
        return None, text
    try:
        code_start  = text.index("```python") + 9
        code_end    = text.index("```", code_start)
        code        = text[code_start:code_end].strip()
        expl_start  = text.index("EXPLANATION:") + 12
        explanation = text[expl_start:].split("\n")[0].strip()
        return code, explanation
    except ValueError:
        return None, text

# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🏀 NBA AI Analyst")
st.markdown(
    "Ask anything about 75 years of NBA data — teams, players, standings, and more. "
    "Powered by Claude + a Random Forest model trained on 1946–2022 games."
)

with st.sidebar:
    st.header("Setup")
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        api_key = st.text_input(
            "Anthropic API key", type="password",
            help="Get one free at console.anthropic.com"
        )
    st.divider()
    st.header("Model stats")
    st.metric("Test accuracy",    "78.1%")
    st.metric("ROC AUC",          "0.854")
    st.metric("Games in dataset", f"{len(game_df):,}")
    st.metric("Player records",   f"{len(player_df):,}")
    st.metric("Seasons covered",  f"{game_df['season'].min()}–{game_df['season'].max()}")
    st.divider()
    st.header("Try asking...")
    examples = [
        "Who scored the most points on April 18 of any year?",
        "What was Kobe Bryant's highest scoring game?",
        "Which team had the best record in 2016?",
        "What team had the best home win rate in 2015?",
        "How has FG% changed over the decades?",
        "Which season had the most games played?",
        "What's the average points scored by winning teams?",
        "Which year saw the biggest drop in home court advantage?",
        "What is the most important stat for predicting wins?",
        "How accurate is the model at predicting losses?",
    ]
    for q in examples:
        if st.button(q, use_container_width=True):
            st.session_state.example_question = q

# ── Chat ─────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "raw_history" not in st.session_state:
    st.session_state.raw_history = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "query" in msg:
            with st.expander("View query"):
                st.code(msg["query"], language="python")
        if "query_result" in msg:
            with st.expander("View raw result"):
                st.text(msg["query_result"])

if "example_question" in st.session_state:
    prompt = st.session_state.pop("example_question")
else:
    prompt = st.chat_input("Ask anything about the NBA data...")

if prompt:
    if not api_key:
        st.error("Please enter your Anthropic API key in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.raw_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                client = anthropic.Anthropic(api_key=api_key)

                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,
                    system=build_schema_context(),
                    messages=st.session_state.raw_history
                )
                first_response = response.content[0].text
                code, explanation = extract_query(first_response)

                if code:
                    with st.spinner(f"Running query: {explanation}"):
                        query_result = execute_query(code)

                    followup_prompt = (
                        f"The query returned this result:\n\n{query_result}\n\n"
                        f"Now answer the user's original question in plain English "
                        f"using these numbers. Be concise and insightful."
                    )
                    followup_history = st.session_state.raw_history + [
                        {"role": "assistant", "content": first_response},
                        {"role": "user",      "content": followup_prompt}
                    ]
                    final_response = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=512,
                        system=build_schema_context(),
                        messages=followup_history
                    )
                    answer = final_response.content[0].text

                    st.write(answer)
                    with st.expander("View query"):
                        st.code(code, language="python")
                    with st.expander("View raw result"):
                        st.text(query_result)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "query": code,
                        "query_result": query_result
                    })
                    st.session_state.raw_history.append({
                        "role": "assistant",
                        "content": answer
                    })

                else:
                    st.write(first_response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": first_response
                    })
                    st.session_state.raw_history.append({
                        "role": "assistant",
                        "content": first_response
                    })

            except anthropic.AuthenticationError:
                st.error("Invalid API key. Check your key at console.anthropic.com.")
            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")

if st.session_state.messages:
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.raw_history = []
        st.rerun()
