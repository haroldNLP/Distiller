from enum import Enum


class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            "%r is not a valid %s, please select one of %s"
            % (value, cls.__name__, str(list(cls._value2member_map_.keys())))
        )


class TruncationStrategy(ExplicitEnum):
    """
    Possible values for the ``truncation`` argument in :meth:`PreTrainedTokenizerBase.__call__`. Useful for
    tab-completion in an IDE.
    """

    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"