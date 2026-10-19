
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from train_ipl_three_predictions import (
    DATA_PATH,
    MODEL_PATH,
    load_models,
    predict_final_three_overs,
    predict_first_innings_score,
    predict_live_match_winner,
)


st.set_page_config(
    page_title="IPL Analytics Hub",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)


TEAM_NAMES = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
]

# Demo profiles keep the visualisations responsive and reproducible. Replace these
# values with a trained model or historical IPL dataset when it is available.
TEAM_STRENGTH = {
    "Chennai Super Kings": 76,
    "Delhi Capitals": 65,
    "Gujarat Titans": 74,
    "Kolkata Knight Riders": 78,
    "Lucknow Super Giants": 70,
    "Mumbai Indians": 75,
    "Punjab Kings": 66,
    "Rajasthan Royals": 73,
    "Royal Challengers Bengaluru": 72,
    "Sunrisers Hyderabad": 71,
}

VENUE_PROFILES = {
    "M. Chinnaswamy Stadium": {"avg": 184, "chase": 56.8, "pace": 57, "boundary": 72},
    "Wankhede Stadium": {"avg": 178, "chase": 55.1, "pace": 61, "boundary": 69},
    "Eden Gardens": {"avg": 176, "chase": 53.4, "pace": 55, "boundary": 66},
    "MA Chidambaram Stadium": {"avg": 164, "chase": 48.9, "pace": 45, "boundary": 53},
    "Narendra Modi Stadium": {"avg": 171, "chase": 51.7, "pace": 58, "boundary": 63},
    "Rajiv Gandhi Intl Stadium": {"avg": 174, "chase": 52.6, "pace": 54, "boundary": 61},
}
VENUES = list(VENUE_PROFILES)
YEARS = list(range(2016, 2024))
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {
            --ink: #ecf6ff;
            --muted: #91a4bd;
            --panel: rgba(17, 30, 52, .78);
            --line: rgba(157, 191, 224, .16);
            --cyan: #28e0c1;
            --blue: #62a8ff;
            --gold: #ffc857;
            --pink: #fa6da4;
        }
        .stApp {
            background:
                radial-gradient(circle at 11% -8%, rgba(40,224,193,.18), transparent 30rem),
                radial-gradient(circle at 92% 5%, rgba(98,168,255,.16), transparent 28rem),
                #07111f;
            color: var(--ink);
            font-family: 'DM Sans', sans-serif;
        }
        [data-testid="stHeader"] { background: rgba(7,17,31,.72); }
        [data-testid="stToolbar"] { right: 1rem; }
        .block-container { max-width: 100% !important; padding: 1.1rem clamp(1rem, 2.6vw, 3rem) 3rem !important; }
        [data-testid="stHorizontalBlock"] { gap: 1.6rem !important; }
        [data-testid="stVerticalBlock"] { gap: 1.35rem; }

        .hero {
            position: relative;
            overflow: hidden;
            background: linear-gradient(115deg, rgba(19,42,68,.96), rgba(14,25,47,.92));
            border: 1px solid rgba(105, 225, 207, .25);
            box-shadow: 0 18px 44px rgba(0, 0, 0, .25);
            border-radius: 24px;
            padding: 26px 34px;
            margin-bottom: 14px;
        }
        .hero:after {
            content: 'IPL'; position: absolute; right: 28px; top: -26px;
            font: 700 128px/1 'Space Grotesk', sans-serif; letter-spacing: -9px;
            color: rgba(142, 191, 255, .06); pointer-events: none;
        }
        .eyebrow { color: var(--cyan); letter-spacing: .16em; font-size: .72rem; font-weight: 700; text-transform: uppercase; }
        .hero h1 { font: 700 clamp(1.9rem, 4vw, 3.15rem)/1.05 'Space Grotesk', sans-serif; margin: .42rem 0 .6rem; color: #f7fbff; }
        .hero h1 span { color: var(--cyan); }
        .hero p { color: #b5c4d8; max-width: 620px; font-size: 1rem; margin: 0; line-height: 1.65; }
        .hero.compact { padding: 16px 30px; border-radius: 20px; }
        .hero.compact h1 { font-size: clamp(1.35rem, 2.3vw, 1.9rem); margin: .2rem 0 0; }
        .hero.compact p { display: none; }
        .hero.compact:after { font-size: 84px; top: -14px; letter-spacing: -6px; }

        .top-nav {
            display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
            margin: 0 0 18px; padding: 8px; border-radius: 14px;
            background: rgba(15,29,50,.72); border: 1px solid var(--line);
        }
        .top-nav a {
            color: #b7c8dc !important; text-decoration: none !important; font-weight: 600;
            border-radius: 9px; padding: 10px 14px; transition: .2s ease;
        }
        .top-nav a:hover { color: #fff !important; background: rgba(255,255,255,.07); transform: translateY(-1px); }
        .top-nav a.active { color: #07111f !important; background: linear-gradient(90deg, var(--cyan), #8ef0e0); box-shadow: 0 4px 15px rgba(40,224,193,.16); }

        .section-kicker { color: var(--cyan); font-size: .75rem; letter-spacing: .15em; font-weight: 700; text-transform: uppercase; margin-bottom: .35rem; }
        h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: #f3f8ff !important; }
        .section-title { font: 600 1.65rem/1.2 'Space Grotesk', sans-serif; color: #f3f8ff; margin: 0 0 1.3rem; }

        .info-card, .feature-card {
            height: 100%; box-sizing: border-box; border: 1px solid var(--line);
            background: var(--panel); border-radius: 18px; padding: 24px;
            box-shadow: 0 10px 28px rgba(0,0,0,.14);
        }
        .feature-icon { font-size: 1.55rem; margin-bottom: .8rem; }
        .feature-card h3, .info-card h3 { margin: 0 0 .55rem; font-size: 1.1rem; }
        .feature-card p, .info-card p { color: var(--muted); line-height: 1.65; margin: 0; }
        .mission-card { border-top: 3px solid var(--cyan); }
        .vision-card { border-top: 3px solid var(--gold); }
        .model-note {
            border-left: 3px solid var(--blue); background: rgba(98,168,255,.08);
            color: #b8cce7; padding: 12px 15px; border-radius: 0 10px 10px 0;
            margin: 4px 0 18px; font-size: .88rem;
        }
        .result-banner {
            border: 1px solid rgba(40,224,193,.22); border-radius: 14px; padding: 15px 18px;
            background: linear-gradient(90deg, rgba(40,224,193,.12), rgba(98,168,255,.07));
            color: #dffdf7; margin: 14px 0 18px;
        }
        .result-banner strong { color: var(--cyan); }
        [data-testid="stMetric"] {
            background: rgba(15,29,50,.7); border: 1px solid var(--line); border-radius: 14px;
            padding: 14px 16px; min-height: 90px;
        }
        [data-testid="stMetricLabel"] { color: var(--muted) !important; }
        [data-testid="stMetricValue"] { color: #f5fbff !important; font-family: 'Space Grotesk', sans-serif; }
        [data-testid="stMetricDelta"] svg { display: none; }
        div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
            background: rgba(15,29,50,.75); border-color: var(--line); border-radius: 10px;
        }
        div[data-baseweb="select"] *, div[data-baseweb="input"] * { color: #eff7ff !important; }
        .stSlider [data-baseweb="slider"] div[role="slider"] { background-color: var(--cyan); }
        .stPlotlyChart { border: 1px solid var(--line); border-radius: 16px; padding: 12px 14px 6px; background: rgba(10,22,40,.43); }
        [data-testid="stCaptionContainer"] { color: var(--muted); }
        @media (max-width: 620px) {
            .hero { padding: 24px; } .top-nav { gap: 3px; padding: 6px; }
            .top-nav a { padding: 8px 9px; font-size: .78rem; } .block-container { padding-top: 1rem; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_current_page() -> str:
    """Read navigation state while retaining compatibility with older Streamlit."""
    try:
        requested_page = st.query_params.get("page", "home")
    except AttributeError:
        requested_page = st.experimental_get_query_params().get("page", ["home"])[0]
    return requested_page if requested_page in PAGE_LABELS else "home"


def render_header(page: str) -> None:
    hero_class = "hero" if page == "home" else "hero compact"
    st.markdown(
        f"""
        <div class="{hero_class}">
            <div class="eyebrow">Data-led cricket intelligence</div>
            <h1>Advanced cricket analytics with machine learning</h1>
            <p>Explore venue patterns, toss context and live scoring scenarios in one focused match-day dashboard.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    nav_items = "".join(
        f'<a class="{"active" if slug == page else ""}" href="?page={slug}" target="_self">{label}</a>'
        for slug, label in PAGE_LABELS.items()
    )
    st.markdown(f'<nav class="top-nav" aria-label="Main navigation">{nav_items}</nav>', unsafe_allow_html=True)


def section_title(kicker: str, title: str) -> None:
    st.markdown(
        f'<div class="section-kicker">{kicker}</div><div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )


def chart_theme(
    fig: go.Figure, *, height: int = 420, title: str | None = None, subtitle: str | None = None
) -> go.Figure:
    """Apply the dashboard look and give the chart a clear title (plus optional subtitle).

    A title already set on the figure (for example by plotly express) is kept unless a new
    one is passed in, so charts never lose their heading.
    """
    heading = title if title is not None else fig.layout.title.text
    fig.update_layout(
        template="plotly_dark",
        height=height,
        font={"family": "DM Sans", "color": "#dce9f7", "size": 13},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 60, "r": 36, "t": 104 if subtitle else 84, "b": 60},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        hoverlabel={"bgcolor": "#132846", "font_color": "#f4fbff"},
        bargap=0.32,
    )
    if heading:
        text = f"<b>{heading}</b>"
        if subtitle:
            text += f"<br><span style='font-size:12px;font-weight:400;color:#91a4bd'>{subtitle}</span>"
        fig.update_layout(
            title={
                "text": text,
                "x": 0.01,
                "xanchor": "left",
                "y": 0.96,
                "yanchor": "top",
                "font": {"family": "Space Grotesk", "size": 18, "color": "#f3f8ff"},
            }
        )
    fig.update_xaxes(gridcolor="rgba(156, 189, 220, .12)", linecolor="rgba(156, 189, 220, .22)", automargin=True)
    fig.update_yaxes(gridcolor="rgba(156, 189, 220, .12)", linecolor="rgba(156, 189, 220, .22)", automargin=True)
    return fig


def add_headroom(fig: go.Figure, top_value: float, factor: float = 1.22) -> go.Figure:
    """Leave space above the tallest bar so its value label is never clipped."""
    fig.update_yaxes(range=[0, max(float(top_value), 1.0) * factor])
    fig.update_traces(cliponaxis=False)
    return fig


@st.cache_resource(show_spinner="Loading trained IPL models...")
def load_live_models() -> dict:
    """Load the three saved XGBoost models once per Streamlit server."""
    return load_models(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_live_input_options(data_path: str) -> tuple[list[str], list[str]]:
    """Read only the input columns needed by the trained live models."""
    live_data = pd.read_csv(data_path, usecols=["team1", "team2", "venue"])
    teams = sorted(set(live_data["team1"].dropna()) | set(live_data["team2"].dropna()))
    venues = sorted(live_data["venue"].dropna().unique())
    return teams, venues


@st.cache_data(show_spinner="Loading 2023–2026 venue data...")
def load_recent_venue_innings(data_path: str) -> pd.DataFrame:
    """Build one real scorecard row per innings for IPL seasons 2023 to 2026."""
    deliveries = pd.read_csv(
        data_path,
        usecols=["match_id", "season", "venue", "innings", "total_runs", "is_wicket", "batting_team_won"],
    )
    deliveries["season"] = pd.to_numeric(deliveries["season"], errors="coerce")
    deliveries = deliveries[
        deliveries["season"].between(2023, 2026)
        & deliveries["venue"].notna()
        & deliveries["innings"].isin([1, 2])
    ].copy()
    return (
        deliveries.groupby(["match_id", "season", "venue", "innings"], as_index=False)
        .agg(
            runs=("total_runs", "sum"),
            wickets=("is_wicket", "sum"),
            batting_team_won=("batting_team_won", "max"),
        )
    )


@st.cache_data(show_spinner=False)
def load_historical_venue_match_counts(data_path: str) -> dict[str, int]:
    """Count every historical IPL match at each venue in the supplied dataset."""
    matches = pd.read_csv(data_path, usecols=["match_id", "venue"]).dropna(subset=["venue"])
    return matches.groupby("venue")["match_id"].nunique().astype(int).to_dict()


def render_chase_model_charts(
    *, result: dict, target: int, current_runs: int, wickets: int, overs: float, title: str
) -> None:
    """Show the trained chase probabilities alongside the live match state."""
    legal_balls = overs_to_balls(overs)
    if legal_balls is None:
        return

    balls_remaining = 120 - legal_balls
    runs_required = max(target - current_runs, 0)
    current_rate = current_runs * 6 / legal_balls
    required_rate = runs_required * 6 / max(balls_remaining, 1)
    chase_probability = float(result["chasing_team_win_probability"])
    defend_probability = float(result["defending_team_win_probability"])

    metrics = st.columns(4)
    metrics[0].metric("Runs required", runs_required, border=True)
    metrics[1].metric("Balls remaining", balls_remaining, border=True)
    metrics[2].metric("Current run rate", f"{current_rate:.2f}", border=True)
    metrics[3].metric("Required run rate", f"{required_rate:.2f}", border=True)

    left, right = st.columns(2)
    with left:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=chase_probability,
                number={"suffix": "%", "font": {"color": "#f4fbff", "size": 44}},
                title={
                    "text": f"{result['chasing_team']}<br><span style='font-size:12px;color:#91a4bd'>chasing team win probability</span>",
                    "font": {"color": "#eaf5ff", "size": 16},
                },
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#91a4bd"},
                    "bar": {"color": "#28e0c1", "thickness": 0.28},
                    "bgcolor": "rgba(255,255,255,.08)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 45], "color": "rgba(250,109,164,.20)"},
                        {"range": [45, 55], "color": "rgba(255,200,87,.18)"},
                        {"range": [55, 100], "color": "rgba(40,224,193,.12)"},
                    ],
                },
            )
        )
        chart_theme(gauge, height=440, title=title, subtitle="Trained model estimate for the chasing team")
        gauge.update_layout(margin={"l": 40, "r": 40, "t": 130, "b": 30})
        st.plotly_chart(gauge, width="stretch", config=PLOTLY_CONFIG)
    with right:
        rates = pd.DataFrame(
            {
                "Rate": ["Current", "Required"],
                "Runs per over": [current_rate, required_rate],
            }
        )
        rate_chart = px.bar(
            rates,
            x="Rate",
            y="Runs per over",
            color="Rate",
            text="Runs per over",
            color_discrete_sequence=["#28e0c1", "#fa6da4"],
        )
        chart_theme(
            rate_chart,
            height=440,
            title="Run-Rate Pressure",
            subtitle=f"Current vs required runs per over · {runs_required} needed off {balls_remaining} balls",
        )
        rate_chart.update_layout(showlegend=False)
        rate_chart.update_traces(texttemplate="%{y:.2f}", textposition="outside", textfont={"color": "#dce9f7"})
        rate_chart.update_xaxes(title="")
        rate_chart.update_yaxes(title="Runs per over", rangemode="tozero")
        add_headroom(rate_chart, max(current_rate, required_rate))
        st.plotly_chart(rate_chart, width="stretch", config=PLOTLY_CONFIG)

    probability_data = pd.DataFrame(
        {
            "Team": [result["chasing_team"], result["defending_team"]],
            "Probability": [chase_probability, defend_probability],
        }
    )
    probability_chart = px.bar(
        probability_data,
        x="Probability",
        y="Team",
        orientation="h",
        color="Team",
        text="Probability",
        color_discrete_sequence=["#28e0c1", "#fa6da4"],
    )
    chart_theme(
        probability_chart,
        height=320,
        title="Win Probability Split",
        subtitle="Trained model estimate for the chasing and defending teams",
    )
    probability_chart.update_layout(showlegend=False)
    probability_chart.update_traces(
        texttemplate="%{x:.1f}%", textposition="outside", textfont={"color": "#dce9f7"}, cliponaxis=False
    )
    probability_chart.update_xaxes(
        title="Win probability", range=[0, 112], tickvals=[0, 20, 40, 60, 80, 100], ticksuffix="%"
    )
    probability_chart.update_yaxes(title="", autorange="reversed")
    st.plotly_chart(probability_chart, width="stretch", config=PLOTLY_CONFIG)


def venue_history(venue: str) -> pd.DataFrame:
    profile = VENUE_PROFILES[venue]
    rng = np.random.default_rng(100 + VENUES.index(venue))
    first = np.rint(rng.normal(profile["avg"] + 3, 7, len(YEARS))).astype(int)
    second = np.rint(rng.normal(profile["avg"] - 5, 7, len(YEARS))).astype(int)
    return pd.DataFrame(
        {
            "Season": YEARS * 2,
            "Average score": np.concatenate([first, second]),
            "Innings": ["1st innings"] * len(YEARS) + ["2nd innings"] * len(YEARS),
        }
    )


def overs_to_balls(overs: float) -> int | None:
    whole_overs = int(overs)
    balls = int(round((overs - whole_overs) * 10))
    if balls > 5:
        return None
    return whole_overs * 6 + balls


def balls_to_overs(balls: int) -> str:
    return f"{balls // 6}.{balls % 6}"


def legal_over_options(start_ball: int, end_ball: int = 119) -> list[str]:
    """Return only valid cricket-over labels, such as 11.0 to 11.5 then 12.0."""
    return [balls_to_overs(ball) for ball in range(start_ball, end_ball + 1)]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def overs_stepper(label: str, *, key: str, min_ball: int, max_ball: int = 119) -> float:
    """Overs input that looks and behaves like the other number inputs.

    The built-in - / + buttons move 0.1 at a time and the callback snaps the result back
    onto a legal cricket over: 11.5 + one step -> 12.0, and 12.0 - one step -> 11.5
    (never 11.6 or 11.9). Returns the value in overs notation, for example 11.2.
    """
    widget_key, prev_key = f"{key}_input", f"{key}_prev"
    low, high = float(balls_to_overs(min_ball)), float(balls_to_overs(max_ball))

    def as_tenths(overs: float) -> int:
        return int(round(overs * 10))

    if not isinstance(st.session_state.get(widget_key), float):
        st.session_state[widget_key] = low
        st.session_state[prev_key] = as_tenths(low)

    def snap_to_legal_over() -> None:
        tenths = as_tenths(st.session_state[widget_key])
        whole, ball = divmod(tenths, 10)
        if ball > 5:
            # + pressed on .5 rolls into the next over; - pressed on .0 (or a typed .6-.9) lands on .5
            if tenths - st.session_state[prev_key] == 1:
                whole, ball = whole + 1, 0
            else:
                ball = 5
        legal_ball = int(clamp(whole * 6 + ball, min_ball, max_ball))
        overs = float(balls_to_overs(legal_ball))
        st.session_state[widget_key] = overs
        st.session_state[prev_key] = as_tenths(overs)

    overs = st.number_input(
        label,
        min_value=low,
        max_value=high,
        step=0.1,
        format="%.1f",
        key=widget_key,
        on_change=snap_to_legal_over,
        help="Cricket notation: 11.5 means 11 overs and 5 balls. Pressing + on 11.5 moves to 12.0.",
    )
    return float(overs)


PAGE_LABELS = {
    "home": "Home",
    "venues": "Ground dashboard",
    "toss": "Toss prediction",
    "score": "Score & match prediction",
}


page = get_current_page()
render_header(page)


if page == "home":
    section_title("Welcome", "Built for smarter IPL match conversations")
    intro_left, intro_right = st.columns([1.35, 1])
    with intro_left:
        st.markdown(
            """
            <div class="info-card">
                <h3>From match data to a clearer story</h3>
                <p>IPL Analytics Hub turns key match signals—venue scoring patterns, toss choices, run rate and wickets—into simple, visual match context. It is designed as an approachable companion for cricket fans, analysts and students learning sports analytics.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with intro_right:
        st.markdown(
            """
            <div class="info-card">
                <h3>What you can explore</h3>
                <p>Compare innings at a venue, see how a toss choice changes an outlook, project a first-innings total and follow a live chase probability.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 18px'></div>", unsafe_allow_html=True)
    mission, vision = st.columns(2)
    with mission:
        st.markdown(
            """
            <div class="feature-card mission-card">
                <div class="feature-icon">🎯</div>
                <h3>Our mission</h3>
                <p>Make cricket data feel useful and understandable by presenting match insights through clean visuals instead of rows of isolated numbers.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with vision:
        st.markdown(
            """
            <div class="feature-card vision-card">
                <div class="feature-icon">🔭</div>
                <h3>Our vision</h3>
                <p>Create a trusted, fan-friendly analytics space where every ball, venue and tactical decision can be explored with confidence.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 26px'></div>", unsafe_allow_html=True)
    section_title("How it works", "Three views. One match story.")
    feature_1, feature_2, feature_3 = st.columns(3)
    for column, icon, title, description in [
        (feature_1, "🏟️", "Ground dashboard", "Study scoring, chasing and wicket-taking patterns for a selected venue."),
        (feature_2, "🪙", "Toss prediction", "Compare the toss context with a visual match outlook for two teams."),
        (feature_3, "📈", "Score & match", "Project the first innings and view live chase pressure with clear charts."),
    ]:
        with column:
            st.markdown(
                f'<div class="feature-card"><div class="feature-icon">{icon}</div><h3>{title}</h3><p>{description}</p></div>',
                unsafe_allow_html=True,
            )


elif page == "venues":
    section_title("Venue intelligence", "Ground dashboard")
    recent_innings = load_recent_venue_innings(str(DATA_PATH))
    active_venues = sorted(recent_innings["venue"].unique())
    selected_venue = st.selectbox(
        "Choose an IPL venue active between 2023 and 2026", active_venues, key="active_venue"
    )
    venue_innings = recent_innings[recent_innings["venue"] == selected_venue].copy()
    historical_match_counts = load_historical_venue_match_counts(str(DATA_PATH))
    first_innings = venue_innings[venue_innings["innings"] == 1]
    second_innings = venue_innings[venue_innings["innings"] == 2]

    historical_matches_hosted = historical_match_counts.get(selected_venue, 0)
    average_first_score = float(first_innings["runs"].mean())
    chasing_win_rate = float(second_innings["batting_team_won"].mean() * 100)
    average_wickets = float(venue_innings["wickets"].mean())

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Historical match count", historical_matches_hosted, border=True)
    metric_2.metric("Average first-innings score", f"{average_first_score:.0f} runs", border=True)
    metric_3.metric("Chasing win rate", f"{chasing_win_rate:.1f}%", border=True)
    metric_4.metric("Average wickets per innings", f"{average_wickets:.1f}", border=True)

    score_by_season = (
        venue_innings.groupby(["season", "innings"], as_index=False)["runs"]
        .mean()
        .rename(columns={"runs": "Average score"})
    )
    score_by_season["Innings"] = score_by_season["innings"].map({1: "First innings", 2: "Second innings"})

    chase_by_season = (
        second_innings.groupby("season", as_index=False)
        .agg(**{"Chasing win rate": ("batting_team_won", "mean"), "Matches": ("match_id", "nunique")})
    )
    chase_by_season["Chasing win rate"] *= 100

    wins = int(second_innings["batting_team_won"].sum())
    chase_outcomes = pd.DataFrame(
        {"Outcome": ["Chase won", "Chase lost"], "Matches": [wins, len(second_innings) - wins]}
    )

    volume_by_season = (
        venue_innings.groupby("season", as_index=False)["match_id"]
        .nunique()
        .rename(columns={"match_id": "Matches"})
    )

    top_left, top_right = st.columns([1.3, 1])
    with top_left:
        score_chart = px.bar(
            score_by_season,
            x="season",
            y="Average score",
            color="Innings",
            barmode="group",
            color_discrete_map={"First innings": "#28e0c1", "Second innings": "#62a8ff"},
        )
        chart_theme(
            score_chart,
            height=440,
            title="Average Score by Season",
            subtitle=f"First vs second innings · {selected_venue}",
        )
        score_chart.update_xaxes(title="Season", dtick=1)
        score_chart.update_yaxes(title="Average runs", rangemode="tozero")
        st.plotly_chart(score_chart, width="stretch", config=PLOTLY_CONFIG)
    with top_right:
        trend_chart = px.line(
            chase_by_season,
            x="season",
            y="Chasing win rate",
            markers=True,
            text="Matches",
            color_discrete_sequence=["#ffc857"],
        )
        chart_theme(
            trend_chart,
            height=440,
            title="Chasing Success Trend",
            subtitle="Share of chases won each season · n = matches",
        )
        trend_chart.update_traces(
            name="Chasing win rate",
            showlegend=False,
            texttemplate="n=%{text}",
            textposition="top center",
        )
        trend_chart.update_xaxes(title="Season", dtick=1)
        trend_chart.update_yaxes(title="Chases won", ticksuffix="%", range=[0, 100])
        st.plotly_chart(trend_chart, width="stretch", config=PLOTLY_CONFIG)

    bottom_left, bottom_right = st.columns([1, 1.3])
    with bottom_left:
        outcome_chart = px.pie(
            chase_outcomes,
            names="Outcome",
            values="Matches",
            hole=0.68,
            color="Outcome",
            color_discrete_map={"Chase won": "#28e0c1", "Chase lost": "#fa6da4"},
        )
        chart_theme(
            outcome_chart,
            height=440,
            title="Second-Innings Outcomes",
            subtitle="Chases won vs lost, 2023–2026",
        )
        outcome_chart.update_traces(textinfo="label+percent", textposition="outside", automargin=True)
        outcome_chart.update_layout(showlegend=False, margin={"l": 60, "r": 60, "t": 110, "b": 40})
        st.plotly_chart(outcome_chart, width="stretch", config=PLOTLY_CONFIG)
    with bottom_right:
        volume_chart = px.bar(
            volume_by_season,
            x="season",
            y="Matches",
            text="Matches",
            color_discrete_sequence=["#62a8ff"],
        )
        chart_theme(
            volume_chart,
            height=440,
            title="Matches Hosted per Season",
            subtitle="IPL games played at this ground",
        )
        volume_chart.update_traces(
            name="Matches hosted",
            showlegend=False,
            textposition="outside",
            textfont={"color": "#dce9f7"},
        )
        volume_chart.update_xaxes(title="Season", dtick=1)
        volume_chart.update_yaxes(title="Matches", rangemode="tozero")
        add_headroom(volume_chart, volume_by_season["Matches"].max())
        st.plotly_chart(volume_chart, width="stretch", config=PLOTLY_CONFIG)


elif page == "venues_legacy":
    section_title("Legacy venue intelligence", "Ground dashboard")
    selected_venue = st.selectbox("Choose a ground", VENUES, label_visibility="visible")
    profile = VENUE_PROFILES[selected_venue]
    venue_average = profile["avg"]

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Typical first-innings score", f"{venue_average + 3} runs", "+3 above venue par")
    metric_2.metric("Chasing win rate", f"{profile['chase']:.1f}%", "bat second success")
    metric_3.metric("Pace wickets", f"{profile['pace']}%", f"Spin {100 - profile['pace']}%")
    metric_4.metric("Boundary index", f"{profile['boundary']}/100", "scoring freedom")
    st.markdown("<div class='model-note'>Demo venue patterns are stable sample data for the interface. Connect your IPL ball-by-ball dataset here to make the insights historical.</div>", unsafe_allow_html=True)

    history = venue_history(selected_venue)
    score_chart = px.bar(
        history,
        x="Season",
        y="Average score",
        color="Innings",
        barmode="group",
        color_discrete_map={"1st innings": "#28e0c1", "2nd innings": "#62a8ff"},
        title="Average innings score by season",
    )
    chart_theme(score_chart, height=365)
    score_chart.update_yaxes(title="Runs", rangemode="tozero")
    score_chart.update_xaxes(dtick=1)
    st.plotly_chart(score_chart, width="stretch", config=PLOTLY_CONFIG)

    chart_left, chart_right = st.columns([1.15, 0.85])
    with chart_left:
        chase_trend = np.clip(
            profile["chase"] + np.array([-4.8, -2.0, 1.7, -1.4, 3.5, 0.8, -2.1, 2.4]),
            40,
            65,
        )
        trend_chart = go.Figure()
        trend_chart.add_trace(
            go.Scatter(
                x=YEARS,
                y=chase_trend,
                mode="lines+markers",
                line={"color": "#ffc857", "width": 3},
                marker={"size": 8, "color": "#ffc857", "line": {"color": "#07111f", "width": 2}},
                fill="tozeroy",
                fillcolor="rgba(255, 200, 87, .12)",
                hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
                name="Chasing win rate",
            )
        )
        chart_theme(trend_chart, height=330, title="Chasing success trend")
        trend_chart.update_yaxes(title="Wins after chasing (%)", range=[35, 70], ticksuffix="%")
        trend_chart.update_xaxes(dtick=1)
        st.plotly_chart(trend_chart, width="stretch", config=PLOTLY_CONFIG)
    with chart_right:
        wicket_chart = go.Figure(
            go.Pie(
                labels=["Pace wickets", "Spin wickets"],
                values=[profile["pace"], 100 - profile["pace"]],
                hole=0.68,
                marker={"colors": ["#fa6da4", "#62a8ff"], "line": {"color": "#0a1628", "width": 3}},
                textinfo="label+percent",
                textposition="outside",
                hovertemplate="%{label}: %{value}%<extra></extra>",
            )
        )
        chart_theme(wicket_chart, height=330, title="Wicket split by bowling type")
        wicket_chart.update_layout(showlegend=False, margin={"l": 12, "r": 12, "t": 52, "b": 12})
        wicket_chart.add_annotation(text=f"<b>{profile['pace']}%</b><br><span style='font-size:11px'>pace</span>", x=0.5, y=0.5, showarrow=False, font={"size": 18, "color": "#f3f8ff"})
        st.plotly_chart(wicket_chart, width="stretch", config=PLOTLY_CONFIG)


elif page == "toss":
    section_title("Toss context", "Toss & match prediction")
    toss_innings = load_recent_venue_innings(str(DATA_PATH))
    toss_venues = sorted(toss_innings["venue"].unique())
    setup_a, setup_b, setup_c, setup_d = st.columns(4)
    with setup_a:
        team_1 = st.selectbox("Team 1", TEAM_NAMES, index=0)
    with setup_b:
        team_2 = st.selectbox("Team 2", [team for team in TEAM_NAMES if team != team_1], index=4)
    with setup_c:
        toss_venue = st.selectbox("Venue", toss_venues, key="toss_venue")
    with setup_d:
        toss_winner = st.selectbox("Toss winner", [team_1, team_2])
        toss_decision = st.selectbox("Toss decision", ["Field first", "Bat first"])

    toss_favours_chase = toss_decision == "Field first"
    team_1_toss_edge = 4.5 if toss_winner == team_1 else -4.5
    venue_chases = toss_innings[(toss_innings["venue"] == toss_venue) & (toss_innings["innings"] == 2)]
    venue_chase_rate = float(venue_chases["batting_team_won"].mean() * 100)
    venue_chase_edge = (venue_chase_rate - 50) * 0.12
    decision_edge = (2.0 + venue_chase_edge) if toss_favours_chase else (-1.0 - venue_chase_edge)
    strength_edge = (TEAM_STRENGTH[team_1] - TEAM_STRENGTH[team_2]) * 0.55
    team_1_probability = clamp(50 + strength_edge + team_1_toss_edge + (decision_edge if toss_winner == team_1 else -decision_edge), 18, 82)
    team_2_probability = 100 - team_1_probability
    predicted_team = team_1 if team_1_probability >= 50 else team_2
    predicted_probability = max(team_1_probability, team_2_probability)

    st.markdown(
        f"<div class='result-banner'>Match outlook: <strong>{html.escape(predicted_team)}</strong> have a <strong>{predicted_probability:.0f}%</strong> model edge under these toss conditions.</div>",
        unsafe_allow_html=True,
    )

    toss_left, toss_right = st.columns([0.85, 1.15])
    with toss_left:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=predicted_probability,
                number={"suffix": "%", "font": {"color": "#f4fbff", "size": 42}},
                title={"text": f"{predicted_team}<br><span style='font-size:12px;color:#91a4bd'>predicted win chance</span>", "font": {"color": "#eaf5ff", "size": 15}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#91a4bd", "tickwidth": 1},
                    "bar": {"color": "#28e0c1", "thickness": 0.28},
                    "bgcolor": "rgba(255,255,255,.08)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 45], "color": "rgba(250,109,164,.20)"},
                        {"range": [45, 55], "color": "rgba(255,200,87,.18)"},
                        {"range": [55, 100], "color": "rgba(40,224,193,.12)"},
                    ],
                },
            )
        )
        chart_theme(
            gauge,
            height=440,
            title="Predicted Win Chance",
            subtitle=f"{toss_winner} won the toss · {toss_decision.lower()}",
        )
        gauge.update_layout(margin={"l": 40, "r": 40, "t": 130, "b": 30})
        st.plotly_chart(gauge, width="stretch", config=PLOTLY_CONFIG)
    with toss_right:
        probability_data = pd.DataFrame({"Team": [team_1, team_2], "Win probability": [team_1_probability, team_2_probability]})
        probability_chart = px.bar(
            probability_data,
            x="Win probability",
            y="Team",
            orientation="h",
            text="Win probability",
            color="Team",
            color_discrete_sequence=["#28e0c1", "#fa6da4"],
        )
        chart_theme(
            probability_chart,
            height=440,
            title="Head-to-Head Win Probability",
            subtitle="Model estimate of each side's chance of winning",
        )
        probability_chart.update_layout(showlegend=False)
        probability_chart.update_traces(texttemplate="%{x:.1f}%", textposition="inside", insidetextfont={"color": "#07111f", "size": 13})
        probability_chart.update_xaxes(range=[0, 100], ticksuffix="%", title="Model win probability")
        probability_chart.update_yaxes(title="")
        st.plotly_chart(probability_chart, width="stretch", config=PLOTLY_CONFIG)

    radar_left, radar_right = st.columns([1, 1])
    with radar_left:
        team_1_signals = [TEAM_STRENGTH[team_1], 63 if toss_winner == team_1 else 44, 70 if toss_favours_chase and toss_winner == team_1 else 52, venue_chase_rate, 67]
        team_2_signals = [TEAM_STRENGTH[team_2], 63 if toss_winner == team_2 else 44, 70 if toss_favours_chase and toss_winner == team_2 else 52, venue_chase_rate, 67]
        axes = ["Team strength", "Toss position", "Decision fit", "Venue chase trend", "Match readiness"]
        radar = go.Figure()
        for team, values, colour in [(team_1, team_1_signals, "#28e0c1"), (team_2, team_2_signals, "#fa6da4")]:
            radar.add_trace(go.Scatterpolar(r=values + [values[0]], theta=axes + [axes[0]], fill="toself", name=team, line={"color": colour, "width": 2}, opacity=.55))
        chart_theme(
            radar,
            height=480,
            title="Match-Condition Comparison",
            subtitle="Strength, toss, decision fit, venue and readiness (0–100)",
        )
        radar.update_layout(
            polar={"bgcolor": "rgba(0,0,0,0)", "radialaxis": {"visible": True, "range": [0, 100], "gridcolor": "rgba(156,189,220,.16)", "tickfont": {"color": "#91a4bd"}}, "angularaxis": {"gridcolor": "rgba(156,189,220,.16)", "tickfont": {"color": "#dce9f7"}}},
            margin={"l": 70, "r": 70, "t": 120, "b": 40},
        )
        st.plotly_chart(radar, width="stretch", config=PLOTLY_CONFIG)
    with radar_right:
        influence = pd.DataFrame(
            {
                "Factor": ["Team form", "Toss outcome", "Decision", "Venue trend", "Baseline balance"],
                "Influence": [abs(strength_edge), abs(team_1_toss_edge), abs(decision_edge), abs(venue_chase_edge), 50],
                "Direction": ["Matchup", "Toss", toss_decision, toss_venue, "Base"],
            }
        )
        influence_chart = px.funnel(
            influence,
            y="Factor",
            x="Influence",
            color="Direction",
            color_discrete_sequence=["#62a8ff", "#28e0c1", "#ffc857", "#fa6da4"],
        )
        chart_theme(
            influence_chart,
            height=480,
            title="What Drives This Outlook",
            subtitle="Relative influence of each factor on the prediction",
        )
        influence_chart.update_layout(showlegend=False)
        influence_chart.update_xaxes(title="Relative model influence")
        st.plotly_chart(influence_chart, width="stretch", config=PLOTLY_CONFIG)


elif page == "score":
    section_title("Live projections", "Score & match prediction")

    if not DATA_PATH.exists() or not MODEL_PATH.exists():
        st.error("The trained live-model files are missing. Keep the data and models folders inside cricket_analysis.")
        st.stop()

    trained_models = load_live_models()
    model_teams, model_venues = load_live_input_options(str(DATA_PATH))
    score_tab, live_tab, final_tab = st.tabs(
        ["First-innings runs", "Live match winner", "Final three overs"]
    )

    with score_tab:
        st.subheader("First-innings final-score projection")
        team_col, opposition_col, venue_col = st.columns(3)
        with team_col:
            batting_team = st.selectbox("Batting team", model_teams, key="trained_score_batting")
        with opposition_col:
            bowling_team = st.selectbox(
                "Bowling team",
                [team for team in model_teams if team != batting_team],
                key="trained_score_bowling",
            )
        with venue_col:
            score_venue = st.selectbox("Venue", model_venues, key="trained_score_venue")

        runs_col, overs_col, wicket_col = st.columns(3)
        with runs_col:
            score_runs = st.number_input(
                "Current runs", min_value=0, max_value=400, value=85, step=1, key="trained_score_runs"
            )
        with overs_col:
            score_overs = overs_stepper("Overs completed", key="trained_score_overs", min_ball=66)
        with wicket_col:
            score_wickets = st.number_input(
                "Wickets fallen", min_value=0, max_value=9, value=2, step=1, key="trained_score_wickets"
            )
        score_submitted = st.button("Project final score", type="primary", key="trained_score_submit")

        if score_submitted:
            try:
                score_result = predict_first_innings_score(
                    trained_models,
                    batting_team=batting_team,
                    bowling_team=bowling_team,
                    venue=score_venue,
                    current_runs=int(score_runs),
                    wickets=int(score_wickets),
                    overs=float(score_overs),
                )
            except ValueError as error:
                st.error(str(error))
            else:
                chart_col, summary_col = st.columns([1.8, 1])
                with summary_col:
                    st.metric("Projected final score", f"{score_result['projected_final_score']} runs", border=True)
                    st.metric(
                        "Likely range",
                        f"{score_result['likely_low']}–{score_result['likely_high']} runs",
                        border=True,
                    )
                    st.markdown(
                        f"<div class='result-banner'><strong>{html.escape(batting_team)}</strong> are projected to finish around <strong>{score_result['projected_final_score']}</strong> runs.</div>",
                        unsafe_allow_html=True,
                    )
                with chart_col:
                    bar_values = [
                        score_runs,
                        score_result["likely_low"],
                        score_result["projected_final_score"],
                        score_result["likely_high"],
                    ]
                    score_chart = go.Figure(
                        go.Bar(
                            x=["Current score", "Likely low", "Model projection", "Likely high"],
                            y=bar_values,
                            marker_color=["#62a8ff", "#ffc857", "#28e0c1", "#fa6da4"],
                            text=bar_values,
                            textposition="outside",
                            textfont={"color": "#dce9f7"},
                        )
                    )
                    chart_theme(
                        score_chart,
                        height=440,
                        title="Projected First-Innings Total",
                        subtitle="Current score against the model's likely range",
                    )
                    score_chart.update_traces(name="Score outlook", showlegend=False)
                    score_chart.update_layout(showlegend=False)
                    score_chart.update_yaxes(title="Runs", rangemode="tozero")
                    add_headroom(score_chart, max(bar_values))
                    st.plotly_chart(score_chart, width="stretch", config=PLOTLY_CONFIG)

    with live_tab:
        st.subheader("Second-innings live win probability")
        team_col, opposition_col, venue_col = st.columns(3)
        with team_col:
            chasing_team = st.selectbox("Chasing team", model_teams, key="trained_live_chasing")
        with opposition_col:
            defending_team = st.selectbox(
                "Defending team",
                [team for team in model_teams if team != chasing_team],
                key="trained_live_defending",
            )
        with venue_col:
            live_venue = st.selectbox("Venue", model_venues, key="trained_live_venue")

        target_col, runs_col, overs_col, wicket_col = st.columns(4)
        with target_col:
            live_target = st.number_input(
                "Winning target", min_value=1, max_value=400, value=181, step=1, key="trained_live_target"
            )
        with runs_col:
            live_runs = st.number_input(
                "Current runs", min_value=0, max_value=400, value=85, step=1, key="trained_live_runs"
            )
        with overs_col:
            live_overs = overs_stepper("Overs completed", key="trained_live_overs", min_ball=48)
        with wicket_col:
            live_wickets = st.number_input(
                "Wickets fallen", min_value=0, max_value=9, value=2, step=1, key="trained_live_wickets"
            )
        live_submitted = st.button("Calculate live probability", type="primary", key="trained_live_submit")

        if live_submitted:
            try:
                live_result = predict_live_match_winner(
                    trained_models,
                    chasing_team=chasing_team,
                    defending_team=defending_team,
                    venue=live_venue,
                    target=int(live_target),
                    current_runs=int(live_runs),
                    wickets=int(live_wickets),
                    overs=float(live_overs),
                )
            except ValueError as error:
                st.error(str(error))
            else:
                st.markdown(
                    f"<div class='result-banner'><strong>{html.escape(str(live_result['chasing_team']))}</strong> have a <strong>{live_result['chasing_team_win_probability']:.1f}%</strong> live win probability.</div>",
                    unsafe_allow_html=True,
                )
                render_chase_model_charts(
                    result=live_result,
                    target=int(live_target),
                    current_runs=int(live_runs),
                    wickets=int(live_wickets),
                    overs=float(live_overs),
                    title="Live Win Probability",
                )

    with final_tab:
        st.subheader("Final-three-overs win probability")
        team_col, opposition_col, venue_col = st.columns(3)
        with team_col:
            final_chasing_team = st.selectbox("Chasing team", model_teams, key="trained_final_chasing")
        with opposition_col:
            final_defending_team = st.selectbox(
                "Defending team",
                [team for team in model_teams if team != final_chasing_team],
                key="trained_final_defending",
            )
        with venue_col:
            final_venue = st.selectbox("Venue", model_venues, key="trained_final_venue")

        target_col, runs_col, overs_col, wicket_col = st.columns(4)
        with target_col:
            final_target = st.number_input(
                "Winning target", min_value=1, max_value=400, value=181, step=1, key="trained_final_target"
            )
        with runs_col:
            final_runs = st.number_input(
                "Current runs", min_value=0, max_value=400, value=155, step=1, key="trained_final_runs"
            )
        with overs_col:
            final_overs = overs_stepper("Overs completed", key="trained_final_overs", min_ball=102)
        with wicket_col:
            final_wickets = st.number_input(
                "Wickets fallen", min_value=0, max_value=9, value=5, step=1, key="trained_final_wickets"
            )
        final_submitted = st.button(
            "Calculate final-overs probability", type="primary", key="trained_final_submit"
        )

        if final_submitted:
            try:
                final_result = predict_final_three_overs(
                    trained_models,
                    chasing_team=final_chasing_team,
                    defending_team=final_defending_team,
                    venue=final_venue,
                    target=int(final_target),
                    current_runs=int(final_runs),
                    wickets=int(final_wickets),
                    overs=float(final_overs),
                )
            except ValueError as error:
                st.error(str(error))
            else:
                st.markdown(
                    f"<div class='result-banner'><strong>{html.escape(str(final_result['chasing_team']))}</strong> have a <strong>{final_result['chasing_team_win_probability']:.1f}%</strong> final-three-overs win probability.</div>",
                    unsafe_allow_html=True,
                )
                render_chase_model_charts(
                    result=final_result,
                    target=int(final_target),
                    current_runs=int(final_runs),
                    wickets=int(final_wickets),
                    overs=float(final_overs),
                    title="Final-Three-Overs Win Probability",
                )


elif page == "score_legacy":
    # Kept only as a non-routed reference while the new trained score page is active.
    section_title("Legacy live projections", "Score & match prediction")
    st.markdown("### First-innings score projection")
    first_a, first_b, first_c, first_d = st.columns(4)
    with first_a:
        batting_team = st.selectbox("Batting team", TEAM_NAMES, index=0)
    with first_b:
        current_runs = st.number_input("Current runs", min_value=0, max_value=400, value=85, step=1)
    with first_c:
        completed_overs = st.number_input("Overs completed", min_value=0.0, max_value=19.5, value=10.0, step=0.1, help="Use cricket notation: .0 to .5 only, for example 10.2.")
    with first_d:
        wickets_lost = st.number_input("Wickets lost", min_value=0, max_value=10, value=2, step=1)

    first_balls = overs_to_balls(completed_overs)
    if first_balls is None:
        st.error("Overs must end in .0, .1, .2, .3, .4 or .5. For example, use 10.5 rather than 10.6.")
    else:
        current_rpo = (current_runs * 6 / first_balls) if first_balls else 0
        venue_par_rpo = VENUE_PROFILES["Wankhede Stadium"]["avg"] / 20
        projected_rpo = clamp((current_rpo * 0.7) + (venue_par_rpo * 0.3) - max(wickets_lost - 3, 0) * 0.25, 5.5, 16.0)
        projected_total = int(round(current_runs + ((120 - first_balls) * projected_rpo / 6)))
        lower_total = max(current_runs, projected_total - 7)
        upper_total = projected_total + 10
        projection_metrics = st.columns(4)
        projection_metrics[0].metric("Projected total", f"{lower_total}–{upper_total}")
        projection_metrics[1].metric("Current run rate", f"{current_rpo:.2f}")
        projection_metrics[2].metric("Projected run rate", f"{projected_rpo:.2f}")
        projection_metrics[3].metric("Balls remaining", 120 - first_balls)
        st.markdown(
            f"<div class='result-banner'><strong>{html.escape(batting_team)}</strong> are projected to finish around <strong>{projected_total}</strong>, based on the current scoring rate and wickets in hand.</div>",
            unsafe_allow_html=True,
        )

        current_over_value = first_balls / 6
        observed_x = np.linspace(0, current_over_value, max(2, int(current_over_value * 5) + 1))
        observed_y = np.linspace(0, current_runs, len(observed_x))
        projected_x = np.linspace(current_over_value, 20, max(2, int((20 - current_over_value) * 5) + 1))
        projected_y = current_runs + (projected_x - current_over_value) * projected_rpo
        innings_chart = go.Figure()
        innings_chart.add_trace(go.Scatter(x=observed_x, y=observed_y, mode="lines", name="Current innings", line={"color": "#28e0c1", "width": 4}, fill="tozeroy", fillcolor="rgba(40,224,193,.12)", hovertemplate="Over %{x:.1f}: %{y:.0f} runs<extra></extra>"))
        innings_chart.add_trace(go.Scatter(x=projected_x, y=projected_y, mode="lines", name="Projected finish", line={"color": "#ffc857", "width": 3, "dash": "dash"}, hovertemplate="Over %{x:.1f}: %{y:.0f} runs<extra></extra>"))
        innings_chart.add_vline(x=current_over_value, line_dash="dot", line_color="#fa6da4", annotation_text=f"Now · {balls_to_overs(first_balls)} overs", annotation_font_color="#fa6da4")
        chart_theme(innings_chart, height=360, title="Innings progression and projected finish")
        innings_chart.update_xaxes(title="Overs", range=[0, 20])
        innings_chart.update_yaxes(title="Cumulative runs", rangemode="tozero")
        st.plotly_chart(innings_chart, width="stretch", config=PLOTLY_CONFIG)

        phase_labels = ["Powerplay\n0–6", "Middle\n7–15", "Death\n16–20"]
        phase_shares = np.array([0.29, 0.43, 0.28])
        phase_projection = np.rint(projected_total * phase_shares).astype(int)
        phase_chart = go.Figure(go.Bar(x=phase_labels, y=phase_projection, marker={"color": ["#62a8ff", "#28e0c1", "#ffc857"], "line": {"width": 0}}, text=phase_projection, textposition="outside", textfont={"color": "#dce9f7"}, hovertemplate="%{x}: %{y} estimated runs<extra></extra>"))
        chart_theme(phase_chart, height=300, title="Estimated scoring by innings phase")
        phase_chart.update_yaxes(title="Projected runs", rangemode="tozero")
        st.plotly_chart(phase_chart, width="stretch", config=PLOTLY_CONFIG)

    st.markdown("<div style='height: 26px'></div>", unsafe_allow_html=True)
    st.markdown("### Second-innings match outlook")
    chase_a, chase_b, chase_c, chase_d = st.columns(4)
    with chase_a:
        target_score = st.slider("First-innings score", min_value=100, max_value=300, value=185)
    with chase_b:
        chasing_team = st.selectbox("Chasing team", TEAM_NAMES, index=5)
    with chase_c:
        defending_team = st.selectbox("Defending team", [team for team in TEAM_NAMES if team != chasing_team], index=0)
    with chase_d:
        chase_runs = st.number_input("Current chase runs", min_value=0, max_value=400, value=45, step=1)
        chase_overs = st.number_input("Chase overs", min_value=0.0, max_value=19.5, value=6.0, step=0.1, help="Use cricket notation: .0 to .5 only.")
        chase_wickets = st.number_input("Chase wickets", min_value=0, max_value=10, value=1, step=1)

    chase_balls = overs_to_balls(chase_overs)
    if chase_balls is None:
        st.error("Chase overs must end between .0 and .5.")
    else:
        target_to_win = target_score + 1
        runs_needed = max(target_to_win - chase_runs, 0)
        balls_left = max(120 - chase_balls, 0)
        current_chase_rpo = (chase_runs * 6 / chase_balls) if chase_balls else 0
        required_rpo = (runs_needed * 6 / balls_left) if balls_left else 0
        if chase_runs >= target_to_win:
            chase_probability = 99.0
        elif balls_left == 0:
            chase_probability = 1.0
        else:
            pace_difference = current_chase_rpo - required_rpo
            chase_probability = clamp(50 + pace_difference * 6 - chase_wickets * 3 + (TEAM_STRENGTH[chasing_team] - TEAM_STRENGTH[defending_team]) * .25, 1, 99)
        defend_probability = 100 - chase_probability
        current_finish = int(round(chase_runs + balls_left * max(current_chase_rpo, 0) / 6))
        pressure_label = "In control" if chase_probability >= 60 else "Balanced contest" if chase_probability >= 43 else "Pressure on chase"

        match_metrics = st.columns(5)
        match_metrics[0].metric("Runs needed", runs_needed)
        match_metrics[1].metric("Balls left", balls_left)
        match_metrics[2].metric("Required rate", f"{required_rpo:.2f}")
        match_metrics[3].metric("Current rate", f"{current_chase_rpo:.2f}")
        match_metrics[4].metric("Match state", pressure_label)
        favoured_team = chasing_team if chase_probability >= 50 else defending_team
        favoured_probability = max(chase_probability, defend_probability)
        st.markdown(
            f"<div class='result-banner'><strong>{html.escape(favoured_team)}</strong> have a <strong>{favoured_probability:.0f}%</strong> live win outlook. The chase currently projects to <strong>{current_finish}</strong> against a target of <strong>{target_to_win}</strong>.</div>",
            unsafe_allow_html=True,
        )

        win_chart, rate_chart = st.columns([.9, 1.1])
        with win_chart:
            win_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=chase_probability,
                    number={"suffix": "%", "font": {"color": "#f4fbff", "size": 40}},
                    title={"text": f"{chasing_team}<br><span style='font-size:12px;color:#91a4bd'>chasing win probability</span>", "font": {"color": "#eaf5ff", "size": 15}},
                    gauge={"axis": {"range": [0, 100], "tickcolor": "#91a4bd"}, "bar": {"color": "#62a8ff", "thickness": .28}, "bgcolor": "rgba(255,255,255,.08)", "borderwidth": 0, "steps": [{"range": [0, 45], "color": "rgba(250,109,164,.20)"}, {"range": [45, 55], "color": "rgba(255,200,87,.18)"}, {"range": [55, 100], "color": "rgba(98,168,255,.14)"}]},
                )
            )
            chart_theme(win_gauge, height=320, title="Live win probability")
            win_gauge.update_layout(margin={"l": 20, "r": 20, "t": 56, "b": 12})
            st.plotly_chart(win_gauge, width="stretch", config=PLOTLY_CONFIG)
        with rate_chart:
            rate_data = pd.DataFrame({"Rate": ["Current run rate", "Required run rate", "Venue par rate"], "Runs per over": [current_chase_rpo, required_rpo, VENUE_PROFILES["Wankhede Stadium"]["avg"] / 20]})
            rate_comparison = px.bar(rate_data, x="Rate", y="Runs per over", color="Rate", text="Runs per over", color_discrete_sequence=["#28e0c1", "#fa6da4", "#ffc857"], title="Run-rate pressure")
            chart_theme(rate_comparison, height=320)
            rate_comparison.update_layout(showlegend=False)
            rate_comparison.update_traces(texttemplate="%{y:.2f}", textposition="outside", textfont={"color": "#dce9f7"})
            rate_comparison.update_yaxes(title="Runs per over", rangemode="tozero")
            st.plotly_chart(rate_comparison, width="stretch", config=PLOTLY_CONFIG)

        progress_chart = go.Figure(
            go.Pie(
                labels=["Runs scored", "Runs remaining"],
                values=[min(chase_runs, target_to_win), max(target_to_win - chase_runs, 0)],
                hole=.7,
                sort=False,
                marker={"colors": ["#28e0c1", "rgba(145,164,189,.20)"], "line": {"color": "#0a1628", "width": 3}},
                textinfo="none",
                hovertemplate="%{label}: %{value} runs<extra></extra>",
            )
        )
        chart_theme(progress_chart, height=320, title="Target progress")
        progress_chart.update_layout(showlegend=True, margin={"l": 20, "r": 20, "t": 55, "b": 20})
        progress_chart.add_annotation(text=f"<b>{chase_runs}</b><br><span style='font-size:11px'>of {target_to_win}</span>", x=.5, y=.5, showarrow=False, font={"size": 20, "color": "#f3f8ff"})
        st.plotly_chart(progress_chart, width="stretch", config=PLOTLY_CONFIG)
