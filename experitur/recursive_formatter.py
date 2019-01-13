from collections.abc import Mapping
from string import Formatter
import regex

# BUG: format strings without matching field are changed
# TODO: Don't leave missing fields in recursive calls

# Pattern from https://stackoverflow.com/a/26386070/1116842
PATTERN = regex.compile("{((?>[^{}]+|(?R))*)}")

# Literal values consist solely of a reference without conversions or format specifications
LITERAL_PATTERN = regex.compile("^{([^{}:!]+)}$")


class RecursiveFormatter(Formatter):
    """
    Apply str.format recursively to treat nested expressions.

    For strings of the form {<field_name>} (without conversions or format specs)
    the literal referenced value is returned. If the regular behavior is
    wanted, an empty format spec can be appended: {<field_name>:}.
    """

    def __init__(self, allow_missing=False):
        self._allow_missing = allow_missing

    def vformat(self, format_string, args, kwargs):
        # Recursively substitute nested format strings and references
        format_string = PATTERN.sub(lambda x: "{" + self.vformat(x[1], args, kwargs) + "}",
                                    format_string)

        # Check if this value is a literal value that should not be converted to a string
        literal = LITERAL_PATTERN.match(format_string)

        if literal is not None:
            key = literal[1]
            if key.isdecimal():
                key = int(key)
            # Return literal value
            value = self.get_value(key, args, kwargs)
            return value

        # Ultimately call Formatter.vformat to perform the actual formatting
        return Formatter.vformat(self, format_string, args, kwargs)

    def get_value(self, key, args, kwargs):
        try:
            return Formatter.get_value(self, key, args, kwargs)
        except KeyError:
            if self._allow_missing:
                return "{{{}}}".format(key)
            raise


class RecursiveDict(Mapping):
    """
    A dict where format expressions in string values are recursively applied.

    See str.format. 
    """

    def __init__(self, mapping, allow_missing=False):
        self._mapping = mapping
        self._formatter = RecursiveFormatter(allow_missing)

    def __getitem__(self, key):
        tmpl = self._mapping[key]
        if isinstance(tmpl, str):
            return self._formatter.vformat(tmpl, (), self)
        return tmpl

    def __iter__(self):
        yield from self._mapping

    def __len__(self):
        return len(self._mapping)

    def __repr__(self):  # pragma: no cover
        return "RecursiveDict({!r})".format(self._mapping)

    def as_dict(self):
        return {k: v for k, v in self.items()}
