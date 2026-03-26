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
    game_df            = pd.read_csv("Dataset/game.csv")
    game_df['win']     = game_df['wl_home'].apply(lambda x: 1 if x == 'W' else 0)
    game_df['season']  = game_df['season_id'].astype(str).str[-4:].astype(int)
    feature_importance = pd.read_csv("tableau_feature_importance.csv")
    win_rate           = pd.read_csv("tableau_win_rate_by_season.csv")
    stats_by_outcome   = pd.read_csv("tableau_stats_by_outcome.csv")
    confusion          = pd.read_csv("tableau_confusion_matrix.csv")
    return game_df, feature_importance, win_rate, stats_by_outcome, confusion

game_df, feature_importance, win_rate, stats_by_outcome, confusion = load_data()

# ── Build schema context for Claude ─────────────────────────────────────────
def build_schema_context():
    cols   = game_df.columns.tolist()
    sample = game_df.head(3).to_string(index=False)
    dtypes = game_df.dtypes.to_string()
    fi     = feature_importance.to_string(index=False)
    wr     = win_rate.to_string(index=False)
    sbo    = stats_by_outcome.to_string(index=False)
    conf   = confusion.to_string(index=False)

    return f"""
You are an expert NBA data analyst with access to a real NBA games dataset and a trained
Random Forest model (78.1% accuracy, 0.854 ROC AUC) that predicts home team wins.

You have access to a pandas DataFrame called `game_df` with {len(game_df):,} rows.

COLUMNS: {cols}

DTYPES:
{dtypes}

SAMPLE ROWS:
{sample}

KEY ENGINEERED COLUMNS:
- `win`: 1 if home team won, 0 if lost
- `season`: 4-digit season year (e.g. 2019)

PRE-COMPUTED SUMMARIES:

Feature importances (from Random Forest):
{fi}

Home win rate by season:
{wr}

Average stats by outcome:
{sbo}

Model confusion matrix (test set, 2020+ seasons):
{conf}

INSTRUCTIONS:
When a user asks a question, you have two options:

OPTION 1 — Answer directly from the summaries above if the answer is already there.

OPTION 2 — Write a pandas query if the question needs raw data exploration.
If you write a query, format it EXACTLY like this:

QUERY:
```python
result = game_df[...some pandas code...]
print(result)
```
EXPLANATION: [one sentence explaining what this query does]

Rules for queries:
- Always assign the final result to a variable called `result`
- Always end with print(result) or print(result.to_string())
- Use only pandas operations on `game_df`
- Keep queries concise and correct
- Do not import anything — pandas (pd) and game_df are already available
- After seeing the query result, provide a clear plain-English answer

If you are answering directly without a query, just answer in plain English under 150 words.
"""

# ── Execute pandas query safely ──────────────────────────────────────────────
def execute_query(code: str) -> str:
    local_vars = {"game_df": game_df, "pd": pd}
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
    "Ask anything about 75 years of NBA data. "
    "Powered by Claude + a Random Forest model trained on 1946–2022 games."
)

with st.sidebar:
    st.header("Setup")
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    st.divider()
    st.header("Model stats")
    st.metric("Test accuracy",    "78.1%")
    st.metric("ROC AUC",          "0.854")
    st.metric("Games in dataset", f"{len(game_df):,}")
    st.metric("Seasons covered",  f"{game_df['season'].min()}–{game_df['season'].max()}")
    st.divider()
    st.header("Try asking...")
    examples = [
        "What team had the best home win rate in 2015?",
        "How has FG% changed over the decades?",
        "Which season had the most games played?",
        "What's the average points scored by winning teams?",
        "Is there a correlation between assists and winning?",
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

                # First call — Claude decides whether to query or answer directly
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

                    # Second call — Claude reads result and answers in plain English
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
