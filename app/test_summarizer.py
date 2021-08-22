import summarizer
import pytest
from unittest.mock import Mock
from exceptions import SummaryError, TextTooShort


def test_remove_multiple_spaces():
    actual = summarizer.remove_multiple_spaces("sample   text  ")
    expected = "sample text"
    assert actual == expected


def test_remove_bracketed_numbers():
    actual = summarizer.remove_bracketed_numbers("sample text [1]")
    expected = "sample text"
    assert actual == expected


def test_fix_floating_periods():
    actual = summarizer.fix_floating_periods("sample text . ")
    expected = "sample text."
    assert actual == expected


def test_fix_floating_commas():
    actual = summarizer.fix_floating_commas("sample , text")
    expected = "sample, text"
    assert actual == expected


def test_replace_symbols():
    actual = summarizer.replace_symbols("•sample™ textŒ")
    expected = "sample text"
    assert actual == expected


def test_smart_summary_should_succeed():
    actual = summarizer.smart_summary(
        text="In order to find the most relevant sentences in text, a graph is constructed where the vertices of the graph represent each sentence in a document and the edges between sentences are based on content overlap, namely by calculating the number of words that 2 sentences have in common. Based on this network of sentences, the sentences are fed into the Pagerank algorithm which identifies the most important sentences. When we want to extract a summary of the text, we can now take only the most important sentences. In order to find relevant keywords, the textrank algorithm constructs a word network. This network is constructed by looking which words follow one another. A link is set up between two words if they follow one another, the link gets a higher weight if these 2 words occur more frequenctly next to each other in the text. On top of the resulting network the Pagerank algorithm is applied to get the importance of each word. The top 1/3 of all these words are kept and are considered relevant. After this, a keywords table is constructed by combining the relevant words together if they appear following one another in the text.",
        ratio=0.20)
    expected = {'ratio': 0.20, "summary": "In order to find relevant keywords, the textrank algorithm constructs a word network."}
    assert actual == expected


def test_smart_summary_should_raise_ratio_then_succeed():
    actual = summarizer.smart_summary(
        text="In order to find the most relevant sentences in text, a graph is constructed where the vertices of the graph represent each sentence in a document and the edges between sentences are based on content overlap, namely by calculating the number of words that 2 sentences have in common. Based on this network of sentences, the sentences are fed into the Pagerank algorithm which identifies the most important sentences. When we want to extract a summary of the text, we can now take only the most important sentences. In order to find relevant keywords, the textrank algorithm constructs a word network. This network is constructed by looking which words follow one another. A link is set up between two words if they follow one another, the link gets a higher weight if these 2 words occur more frequenctly next to each other in the text. On top of the resulting network the Pagerank algorithm is applied to get the importance of each word. The top 1/3 of all these words are kept and are considered relevant. After this, a keywords table is constructed by combining the relevant words together if they appear following one another in the text.",
        ratio=0.10)
    expected = {'ratio': 0.12, "summary": "In order to find relevant keywords, the textrank algorithm constructs a word network."}
    assert actual == expected


def test_smart_summary_should_raise_ratio_then_raise_exception():
    smart_summary = Mock()
    smart_summary.side_effect = TextTooShort
    with pytest.raises(TextTooShort, match="The provided text is too short to be summarized."):
        summarizer.smart_summary(
            text="In order to find the most relevant sentences in text, a graph is constructed.",
            ratio=0.20)

