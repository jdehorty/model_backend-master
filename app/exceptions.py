class SummaryError(Exception):
    """A base class for summarizer related exceptions."""


class TextTooShort(SummaryError):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.summary = kwargs.get('summary')
        self.ratio = kwargs.get('ratio')
