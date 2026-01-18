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

class TranslateError(Exception):
    pass

class MissingKeysError(TranslateError):
    """Raise when a user has incomplete keys for azure translate"""
    pass

class ExtraKeysError(TranslateError):
    """Raise when a user has extra keys for azure translate"""
    pass


class TagError(Exception):
    """Use this error for errors found in Tagger.py"""
    pass

class FileNameError(TagError):
    """raise when the user supplied a file name that does not start with:
    - google
    - azure
    - deepl
    - opus
    """
    pass