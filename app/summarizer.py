# %%
import re
from summa.summarizer import summarize
from exceptions import SummaryError, TextTooShort


def remove_multiple_spaces(text):
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def remove_bracketed_numbers(text):
    return re.sub(r'\[\d+[, \-\d]{0,5}\d*]', "", text).strip()


def fix_floating_periods(text):
    return re.sub(r'\s\.\s?', ". ", text).strip()


def fix_floating_commas(text):
    return re.sub(r'\s,\s', ", ", text).strip()


def replace_symbols(text):
    text = text.replace('™', '')
    text = text.replace('Œ', '')
    return text.replace('•', '').strip()


def smart_summary(text, ratio):
    summary_text = summarize(text, ratio=ratio)
    ratio, summary_text = raise_ratio(ratio, summary_text, text)
    if summary_text == "":
        raise TextTooShort("The provided text is too short to be summarized.", ratio)
    return {"ratio": float(ratio), "summary": summary_text}


def raise_ratio(ratio, summary_text, text):
    # gradually raises the user's input ratio until a minimal summary is observed
    while summary_text == "" and ratio < 1.0:
        ratio += 0.01
        summary_text = summarize(text, ratio=ratio)
    return round(float(ratio), 2) if float(ratio) < 1.0 else 1.0, summary_text

