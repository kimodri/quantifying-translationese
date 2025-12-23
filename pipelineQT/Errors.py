class ProccesError(Exception):
    """Use this error for errors found in Processor.py."""
    pass

class NoDatasetError(ProccesError):
    """Raise when no datasets is passed."""
    pass

class IncorrectDatasetError(ProccesError):
    """Raise when unidentified dataset/s are supplied."""
    pass

class UnexpectedFileError(ProccesError):
    """Raise when the file suplied to the cleaner is unexpected"""
    pass