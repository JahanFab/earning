"""
Earnings Call Analyzer
Runs NLP on earnings call transcripts to score sentiment and predict
post-earnings price direction using multiple text-analysis approaches.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings

warnings.filterwarnings("ignore")

import re
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


#  Financial Lexicons 

POSITIVE_WORDS = {
    "growth", "record", "exceeded", "strong", "robust", "momentum", "positive",
    "confident", "increase", "gain", "outperform", "expand", "improve", "beat",
    "accelerate", "opportunity", "innovative", "optimistic", "milestone", "success",
    "efficient", "profitability", "margin", "revenue", "upside", "exceeded",
    "delivered", "achieved", "outstanding", "solid", "impressive", "beat",
    "higher", "raised", "above", "ahead", "excellent", "exceptional",
}

NEGATIVE_WORDS = {
    "decline", "decrease", "miss", "weak", "challenge", "difficult", "headwind",
    "concern", "uncertainty", "risk", "pressure", "loss", "below", "disappoint",
    "slowdown", "contraction", "lower", "reduce", "cut", "restructure", "impair",
    "volatile", "cautious", "macro", "slower", "missed", "fell", "dropped",
    "declined", "decreased", "reduced", "shortfall", "underperform", "negatively",
}

UNCERTAINTY_WORDS = {
    "uncertain", "unclear", "depends", "potentially", "may", "might", "could",
    "estimate", "approximately", "roughly", "expect", "anticipate", "hope",
    "believe", "assume", "possible", "potential", "guidance", "outlook",
}

FORWARD_LOOKING_WORDS = {
    "next", "future", "upcoming", "quarter", "year", "guidance", "outlook",
    "expect", "forecast", "project", "anticipate", "plan", "target", "goal",
    "pipeline", "roadmap", "strategy", "initiative",
}


#  Sample Earnings Transcripts 
# Realistic synthetic transcripts for demonstration (real pipeline uses SEC EDGAR)

SAMPLE_TRANSCRIPTS = {
    "AAPL_Q1_2024": {
        "ticker": "AAPL",
        "date": "2024-02-01",
        "period": "Q1 2024",
        "actual_direction": 1,   # +1 price went up post-earnings, -1 down
        "transcript": """
Good afternoon. Thank you for joining Apple's first quarter fiscal 2024 earnings call.

We are pleased to report record revenue of $119.6 billion, up 2% year over year,
and earnings per diluted share of $2.18, an all-time record. Our Services business
achieved an all-time revenue record of $23.1 billion, up 11% year over year.

iPhone revenue came in at $69.7 billion, growing year over year for the quarter.
We set revenue records in many markets including Canada, Spain, and India.
The install base of active devices reached an all-time high across all products.

We are optimistic about the opportunities ahead. Our innovation pipeline remains
strong, and we believe our ecosystem continues to create incredible value for customers.
Customer satisfaction for iPhone remains exceptionally high at 98 percent.

Regarding guidance for Q2, we expect revenue to be similar to Q1 on a reported basis.
Services revenue should grow double digits year over year. We remain confident in
our ability to execute and deliver value to shareholders. Free cash flow generation
was $39.9 billion for the quarter, and we returned over $27 billion to shareholders.

Our gross margin of 45.9% exceeded our guidance range. We are raising our quarterly
dividend and authorizing $90 billion in additional share repurchases. We are very
confident in our long-term business trajectory.
        """,
    },
    "META_Q2_2023": {

        "ticker": "META",
        "date": "2023-07-26",
        "period": "Q2 2023",
        "actual_direction": 1,
        "transcript": """
Thank you for joining Meta's second quarter 2023 earnings call.

Revenue was $32.0 billion, up 11% year over year. This was ahead of our expectations
and reflects the strong performance of our advertising business. Daily active people
reached 3.07 billion, a new record. Net income was $7.79 billion, up 16%.

Our AI investments are delivering real results. The efficiency of our ad targeting
has improved significantly, and we're seeing strong ROI across our family of apps.
Reels continues to gain momentum and is now a meaningful revenue contributor.

We are excited about the opportunities ahead. Our technical infrastructure is
becoming more efficient through our year of efficiency initiatives. We've made
substantial progress on cost discipline while investing in future growth.

For Q3, we expect revenue in the range of $32-34.5 billion. We remain optimistic
about the remainder of 2023 and beyond. Our AI roadmap is progressing well, and
we believe we are well-positioned to benefit from the next wave of technology.

The team has executed exceptionally well, and I'm confident in our ability to
continue delivering strong results. Operating margin expanded significantly this
quarter, reflecting improved operational discipline.
        """,
    },
    "INTC_Q3_2022": {

        "ticker": "INTC",
        "date": "2022-10-27",
        "period": "Q3 2022",
        "actual_direction": -1,
        "transcript": """
Good afternoon. Intel's third quarter 2022 results reflect a challenging environment.

Revenue was $15.3 billion, down 20% year over year, missing our prior guidance.
The PC market deterioration has been significantly worse than expected. Customer
inventory corrections are proving more severe and prolonged than we anticipated.

We are taking immediate action to reduce costs. We expect to cut approximately
$3 billion in spending by end of 2023, rising to $8 to $10 billion by 2025.
Gross margin declined to 45.9%, below our earlier guidance of approximately 49%.

Our data center and AI business also faced headwinds, declining year over year
due to weakness in enterprise and government spending. We are disappointed with
these results and recognize the need to address our competitive position.

For Q4, we expect revenue of approximately $14 to $15 billion, below seasonal norms.
The macro environment remains uncertain and we see continued customer inventory
adjustments. We acknowledge the challenges in our execution and are working to
improve. Our process technology roadmap remains on track, but near-term headwinds
will persist. We have reduced our full year guidance significantly.

We are cautious on the near-term outlook given the difficult macro environment,
PC market weakness, and enterprise spending softness. Profitability will remain
under pressure through at least the first half of next year.
        """,
    },
    "NVDA_Q2_2024": {

        "ticker": "NVDA",
        "date": "2023-08-23",
        "period": "Q2 FY2024",
        "actual_direction": 1,
        "transcript": """
Good afternoon and welcome to NVIDIA's second quarter fiscal 2024 earnings call.

Revenue was a record $13.51 billion, up 88% from Q1 and up 101% year over year.
This extraordinary performance was driven by exceptional demand for our Data Center
products, particularly our H100 GPU for AI training and inference workloads.

Data center revenue reached $10.32 billion, a new record, up 141% sequentially.
The demand from cloud service providers, consumer internet companies, and enterprises
continues to exceed our supply. We are working aggressively to increase supply.

We delivered record earnings per diluted share of $2.70, up 854% year over year.
Our gross margin of 70.1% exceeded expectations, reflecting the premium positioning
of our AI compute solutions.

For Q3, we expect revenue of approximately $16 billion, plus or minus 2%. We are
very confident in sustained demand throughout the remainder of fiscal 2024 and beyond.
The generative AI wave is in its early innings, and our platform is ideally positioned.

CUDA ecosystem lock-in remains extremely strong. Our networking business through
InfiniBand and Ethernet for AI is also growing rapidly. Customer satisfaction is
exceptional. We see a significant multi-year growth opportunity ahead as enterprises
accelerate their AI infrastructure buildout. This is a once-in-a-generation inflection.
        """,
    },
    "WMT_Q4_2022": {
        "ticker": "WMT",
        "date": "2023-02-21",
        "period": "Q4 FY2023",
        "actual_direction": -1,
        "transcript": """
Thank you for joining Walmart's fourth quarter and fiscal year 2023 earnings call.

Net sales for Q4 increased 4.3% to $164 billion. For the full year, net sales
increased 6.7% to $611 billion. While we grew revenue, profitability was pressured
by elevated inventory shrinkage and markdowns necessary to clear excess merchandise.

Gross profit rate declined 42 basis points for the quarter and 132 basis points
for the year. Operating expenses grew faster than sales, reflecting investments
in associates and technology. We are disappointed that profit growth lagged sales.

For fiscal year 2024, we expect net sales growth of 2.5 to 3%. However, we anticipate
operating income growth will be slightly below sales growth. The consumer environment
remains uncertain, with customers remaining value-focused and selective in spending.

We expect continued pressure on general merchandise categories. Higher-income customers
have been trading down to Walmart, which is positive for traffic but challenges our
margin mix. Grocery inflation has moderated but remains elevated.

Our guidance calls for EPS of $5.90 to $6.05, which is below current consensus.
We are cautious on discretionary spending trends and acknowledge that inventory
management remains a work in progress. The macro environment presents headwinds.
Capital expenditure will be elevated as we invest in supply chain and technology.
        """,
    },
}


#  NLP Sentiment Scoring 

@dataclass
class SentimentScore:
    positive: float = 0.0
    negative: float = 0.0
    uncertainty: float = 0.0
    forward_looking: float = 0.0
    net_sentiment: float = 0.0
    tone_ratio: float = 0.0        # positive / (positive + negative)
    word_count: int = 0
    sentences: int = 0


def tokenize(text: str) -> List[str]:
    """Basic whitespace + punctuation tokenizer."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return [w for w in text.split() if len(w) > 2]


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 10]




def lexicon_sentiment(text: str) -> SentimentScore:
    """Loughran-McDonald style financial lexicon scoring."""
    tokens = tokenize(text)
    sentences = split_sentences(text)
    n = len(tokens)
    if n == 0:
        return SentimentScore()

    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    unc = sum(1 for t in tokens if t in UNCERTAINTY_WORDS)
    fwd = sum(1 for t in tokens if t in FORWARD_LOOKING_WORDS)

    net = (pos - neg) / n
    tone = pos / (pos + neg) if (pos + neg) > 0 else 0.5

    return SentimentScore(
        positive=pos / n,
        negative=neg / n,
        uncertainty=unc / n,
        forward_looking=fwd / n,
        net_sentiment=net,
        tone_ratio=tone,
        word_count=n,
        sentences=len(sentences),
    )


def sentence_level_sentiment(text: str) -> pd.DataFrame:
    """Score every sentence individually — useful for arc analysis."""
    sentences = split_sentences(text)
    rows = []
    for i, sent in enumerate(sentences):
        tokens = tokenize(sent)
        if not tokens:
            continue
        n = len(tokens)
        pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
        neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
        rows.append({
            "sentence_idx": i,
            "sentence": sent[:80],
            "net": (pos - neg) / n,
            "positive": pos / n,
            "negative": neg / n,
            "n_words": n,
        })
    return pd.DataFrame(rows)


#  Feature Extraction 

def extract_guidance_tone(text: str) -> Dict[str, float]:
    """Extract guidance-specific sentiment (sentences containing guidance keywords)."""
    guidance_kw = {"guidance", "expect", "outlook", "forecast", "quarter", "year", "anticipate"}
    sentences = split_sentences(text)
    guidance_sents = [s for s in sentences if any(k in s.lower() for k in guidance_kw)]
    if not guidance_sents:
        return {"guidance_sentiment": 0.0, "guidance_sentence_count": 0}
    combined = " ".join(guidance_sents)
    sc = lexicon_sentiment(combined)
    return {
        "guidance_sentiment": sc.net_sentiment,
        "guidance_sentence_count": len(guidance_sents),
    }


def extract_management_hedging(text: str) -> float:
    """Ratio of hedging/uncertainty language — high = management less confident."""
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    hedge_words = UNCERTAINTY_WORDS | {"approximately", "roughly", "about", "around", "suggest"}
    return sum(1 for t in tokens if t in hedge_words) / len(tokens)


def extract_numeric_mentions(text: str) -> Dict[str, int]:
    """Count mentions of key financial metrics."""
    patterns = {
        "revenue_mentions": r"\brevenue\b",
        "margin_mentions": r"\bmargin\b",
        "growth_mentions": r"\bgrowth\b",
        "guidance_raises": r"\brais(ed|ing)\b",
        "guidance_cuts": r"\b(cut|lower|reduc)\w*\b",
        "record_mentions": r"\brecord\b",
        "miss_mentions": r"\bmiss(ed)?\b",
        "beat_mentions": r"\b(beat|exceeded|above)\b",
    }
    result = {}
    for key, pat in patterns.items():
        result[key] = len(re.findall(pat, text.lower()))
    return result


def compute_features(transcript_data: dict) -> Dict:
    text = transcript_data["transcript"]
    score = lexicon_sentiment(text)
    guidance = extract_guidance_tone(text)
    hedging = extract_management_hedging(text)
    numerics = extract_numeric_mentions(text)

    # Sentiment arc: compare first half vs second half
    sentences = split_sentences(text)
    mid = len(sentences) // 2
    first_half = " ".join(sentences[:mid])
    second_half = " ".join(sentences[mid:])
    sc1 = lexicon_sentiment(first_half)
    sc2 = lexicon_sentiment(second_half)
    sentiment_drift = sc2.net_sentiment - sc1.net_sentiment

    return {
        "ticker": transcript_data["ticker"],
        "period": transcript_data["period"],
        "date": transcript_data["date"],
        # Core sentiment
        "net_sentiment": score.net_sentiment,
        "tone_ratio": score.tone_ratio,
        "positive_density": score.positive,
        "negative_density": score.negative,
        "uncertainty_density": score.uncertainty,
        "forward_looking_density": score.forward_looking,
        # Management quality signals
        "hedging_ratio": hedging,
        "sentiment_drift": sentiment_drift,       # forward vs backward looking tone
        "guidance_sentiment": guidance["guidance_sentiment"],
        # Numeric signals
        **numerics,
        # Ground truth
        "actual_direction": transcript_data.get("actual_direction", 0),
    }


#  Prediction Model 

def build_feature_matrix(feature_list: List[Dict]) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.DataFrame(feature_list)
    y = df["actual_direction"].values
    feature_cols = [
        "net_sentiment", "tone_ratio", "positive_density", "negative_density",
        "uncertainty_density", "forward_looking_density", "hedging_ratio",
        "sentiment_drift", "guidance_sentiment", "revenue_mentions",
        "growth_mentions", "guidance_raises", "guidance_cuts",
        "record_mentions", "beat_mentions", "miss_mentions",
    ]
    X = df[feature_cols].fillna(0)
    return X, y, df


def rule_based_predictor(features: Dict) -> Tuple[int, float, str]:
    """
    Interpretable rule-based classifier (no ML training needed with 5 samples).
    Returns (predicted_direction, confidence, reason).
    """
    score = 0.0
    reasons = []

    if features["net_sentiment"] > 0.01:
        score += 2.0
        reasons.append("positive net sentiment")
    elif features["net_sentiment"] < -0.005:
        score -= 2.0

        reasons.append("negative net sentiment")

    if features["tone_ratio"] > 0.65:
        score += 1.5
        reasons.append("strong positive tone ratio")
    elif features["tone_ratio"] < 0.40:
        score -= 1.5
        reasons.append("weak tone ratio")

    if features["guidance_sentiment"] > 0.005:
        score += 2.0

        reasons.append("positive guidance language")
    elif features["guidance_sentiment"] < -0.005:
        score -= 2.0
        reasons.append("negative guidance language")


    if features["guidance_raises"] > 0:
        score += 1.5
        reasons.append("guidance raised")
    if features["guidance_cuts"] > features["guidance_raises"]:
        score -= 1.5
        reasons.append("guidance cut")


    if features["record_mentions"] >= 3:
        score += 1.0
        reasons.append(f"{features['record_mentions']} 'record' mentions")
    if features["beat_mentions"] >= 2:
        score += 1.0
        reasons.append("beat/exceeded language")
    if features["miss_mentions"] >= 2:
        score -= 1.5
        reasons.append("miss language present")


    if features["sentiment_drift"] > 0:
        score += 0.5
        reasons.append("sentiment improves toward end")
    else:
        score -= 0.5

        reasons.append("sentiment weakens toward end")

    if features["hedging_ratio"] > 0.06:
        score -= 0.5
        reasons.append("high management hedging")



    direction = 1 if score > 0 else -1
    confidence = min(abs(score) / 8.0, 1.0)
    return direction, confidence, "; ".join(reasons[:4])


#  Post-Earnings Price Analysis 

def fetch_post_earnings_return(ticker: str, earnings_date: str, days: int = 5) -> Optional[float]:
    """Fetch actual stock return in the N days after earnings."""
    try:
        start = pd.to_datetime(earnings_date)
        end = start + timedelta(days=days + 10)
        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"),
                         auto_adjust=True, progress=False)
        df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
        if len(df) < 3:
            return None
        ret = (df["Close"].iloc[min(days, len(df)-1)] / df["Close"].iloc[1] - 1)
        return float(ret)
    except Exception:
        return None







#  Visualization 


def plot_sentiment_arc(transcript_data: dict, save_path: str = None):
    text = transcript_data["transcript"]
    sent_df = sentence_level_sentiment(text)
    if sent_df.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(
        f"Sentiment Arc — {transcript_data['ticker']} {transcript_data['period']}",
        fontsize=13, fontweight="bold"
    )





    # Smoothed net sentiment across sentences
    ax = axes[0]
    window = max(3, len(sent_df) // 10)
    smoothed = sent_df["net"].rolling(window, min_periods=1).mean()
    ax.fill_between(sent_df["sentence_idx"], smoothed, 0,
                    where=smoothed >= 0, alpha=0.4, color="#2ecc71", label="Positive")
    ax.fill_between(sent_df["sentence_idx"], smoothed, 0,
                    where=smoothed < 0, alpha=0.4, color="#e74c3c", label="Negative")
    ax.plot(sent_df["sentence_idx"], smoothed, color="navy", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Net Sentiment")
    ax.set_title("Sentence-Level Sentiment Arc (smoothed)")
    ax.legend(fontsize=8)

    # Pos vs Neg density bar chart

    ax2 = axes[1]
    x = sent_df["sentence_idx"]
    ax2.bar(x, sent_df["positive"], color="#2ecc71", alpha=0.6, label="Positive density")
    ax2.bar(x, -sent_df["negative"], color="#e74c3c", alpha=0.6, label="Negative density")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Sentence #")
    ax2.set_ylabel("Word Density")
    ax2.set_title("Positive / Negative Word Density per Sentence")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")
    plt.show()


def plot_comparison_dashboard(results: List[Dict], save_path: str = None):
    df = pd.DataFrame(results)
    n = len(df)
    label = df["ticker"] + "\n" + df["period"]

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Earnings Call NLP Comparison Dashboard", fontsize=14, fontweight="bold")

    colors = ["#2ecc71" if d == 1 else "#e74c3c" for d in df["actual_direction"]]



    # 1. Net sentiment
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(range(n), df["net_sentiment"], color=colors, alpha=0.8)
    ax1.set_xticks(range(n)); ax1.set_xticklabels(label, fontsize=7)
    ax1.set_title("Net Sentiment Score"); ax1.axhline(0, color="black", lw=0.8)


    # 2. Tone ratio
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(n), df["tone_ratio"], color=colors, alpha=0.8)
    ax2.axhline(0.5, color="black", lw=0.8, linestyle="--")
    ax2.set_xticks(range(n)); ax2.set_xticklabels(label, fontsize=7)
    ax2.set_title("Tone Ratio (pos/(pos+neg))")
    ax2.set_ylim(0, 1)

    # 3. Guidance sentiment
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(range(n), df["guidance_sentiment"], color=colors, alpha=0.8)
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_xticks(range(n)); ax3.set_xticklabels(label, fontsize=7)
    ax3.set_title("Guidance Sentiment")


    # 4. Positive vs negative density stacked
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(range(n), df["positive_density"], color="#2ecc71", alpha=0.7, label="Positive")
    ax4.bar(range(n), -df["negative_density"], color="#e74c3c", alpha=0.7, label="Negative")
    ax4.axhline(0, color="black", lw=0.8)
    ax4.set_xticks(range(n)); ax4.set_xticklabels(label, fontsize=7)
    ax4.set_title("Positive / Negative Density"); ax4.legend(fontsize=7)
 # 5. Prediction correctness
    ax5 = fig.add_subplot(gs[1, 1])
    pred_correct = [1 if r["predicted_direction"] == r["actual_direction"] else 0 for r in results]
    bar_colors = ["#2ecc71" if c else "#e74c3c" for c in pred_correct]
    ax5.bar(range(n), df["confidence"], color=bar_colors, alpha=0.8)
    ax5.set_xticks(range(n)); ax5.set_xticklabels(label, fontsize=7)
    ax5.set_title("Prediction Confidence (green=correct)")
    ax5.set_ylim(0, 1)

    # 6. Hedging ratio
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.bar(range(n), df["hedging_ratio"], color="#8e44ad", alpha=0.8)
    ax6.set_xticks(range(n)); ax6.set_xticklabels(label, fontsize=7)
    ax6.set_title("Management Hedging Ratio")

    # Legend


    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#2ecc71", label="Stock Up post-earnings"),
                       Patch(facecolor="#e74c3c", label="Stock Down post-earnings")]
    fig.legend(handles=legend_elements, loc="lower right", fontsize=9)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")
    plt.show()








#  Main 


def run(plot: bool = True):
    print("\n" + "="*60)
    print("  Earnings Call Analyzer — NLP Sentiment & Direction Prediction")
    print("="*60 + "\n")

    print("► Extracting features from transcripts …\n")
    all_features = []
    results = []

    for key, transcript_data in SAMPLE_TRANSCRIPTS.items():
        features = compute_features(transcript_data)
        direction, confidence, reason = rule_based_predictor(features)
        actual = transcript_data.get("actual_direction", 0)

        correct = "CORRECT" if direction == actual else "WRONG  "
        print(f"  [{correct}] {transcript_data['ticker']} {transcript_data['period']}")
        print(f"    Predicted: {'UP' if direction == 1 else 'DOWN'} (confidence {confidence:.0%})")
        print(f"    Actual:    {'UP' if actual == 1 else 'DOWN'}")
        print(f"    Drivers:   {reason}")
        print(f"    Sentiment: net={features['net_sentiment']:.4f}, "
              f"tone={features['tone_ratio']:.2f}, "
              f"guidance={features['guidance_sentiment']:.4f}")
        print()

        features["predicted_direction"] = direction
        features["confidence"] = confidence
        features["reason"] = reason
        all_features.append(features)
        results.append(features)


# Accuracy summary
    correct_count = sum(1 for r in results if r["predicted_direction"] == r["actual_direction"])
    print(f"{'─'*60}")
    print(f"  Accuracy: {correct_count}/{len(results)} = {correct_count/len(results):.0%}")
    print(f"{'─'*60}\n")

    # Feature correlation table
    feat_df = pd.DataFrame(all_features)
    key_cols = ["net_sentiment", "tone_ratio", "guidance_sentiment",
                "hedging_ratio", "sentiment_drift", "actual_direction"]
    print("► Feature Correlation with actual_direction:")
    corr = feat_df[key_cols].corr()["actual_direction"].drop("actual_direction")
    for col, val in corr.items():
        print(f"  {col:35s}: {val:+.3f}")
    print()

    if plot:
        print("► Generating plots …")
        # Sentiment arc for each transcript
        for key, transcript_data in SAMPLE_TRANSCRIPTS.items():
            plot_sentiment_arc(
                transcript_data,
                save_path=f"sentiment_arc_{transcript_data['ticker']}_{transcript_data['period'].replace(' ','_')}.png"
            )
        # Comparison dashboard
        plot_comparison_dashboard(
            results,
            save_path="earnings_comparison_dashboard.png"
        )

    return results


if __name__ == "__main__":
    run(plot=True)
