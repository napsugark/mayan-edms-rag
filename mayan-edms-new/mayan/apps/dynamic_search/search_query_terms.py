from .literals import (
    TERM_MARKER_QUOTE, TERM_MARKER_RAW, TERM_MARKER_SPACE_CHARACTER,
    TERM_OPERATORS
)


class QueryTerm:
    @staticmethod
    def do_query_parse(query):
        result = {}

        for key, value in query.items():
            result[key] = QueryToken.do_query_parse(value=value)

        return result


class QueryToken:
    @staticmethod
    def do_query_parse(value):
        inside_quotes = False
        inside_raw = False
        result = []
        token_letters = []

        for letter in value:
            if not inside_quotes and letter == TERM_MARKER_RAW:
                if inside_raw:
                    inside_raw = False
                    token_string = ''.join(token_letters)
                    result.append(
                        QueryToken(
                            is_meta=False, is_quoted=inside_quotes,
                            is_raw=True, text=token_string
                        )
                    )
                    token_letters = []
                else:
                    inside_raw = True
            elif not inside_raw and letter == TERM_MARKER_QUOTE:
                if inside_quotes:
                    token_string = ''.join(token_letters)
                    result.append(
                        QueryToken(
                            is_meta=False, is_quoted=True, is_raw=False,
                            text=token_string
                        )
                    )
                    token_letters = []
                    inside_quotes = False
                else:
                    inside_quotes = True
            elif not inside_quotes and not inside_raw and letter == TERM_MARKER_SPACE_CHARACTER:
                if token_letters:
                    token_string = ''.join(token_letters)
                    if token_string in TERM_OPERATORS:
                        is_meta = True
                    else:
                        is_meta = False

                    result.append(
                        QueryToken(
                            is_meta=is_meta, is_quoted=False,
                            is_raw=False, text=token_string
                        )
                    )
                    token_letters = []
            else:
                token_letters.append(letter)

        token_string = ''.join(token_letters)

        if token_string and not inside_quotes and not inside_raw:
            result.append(
                QueryToken(
                    is_meta=False, is_quoted=False, is_raw=False,
                    text=token_string
                )
            )

        return result

    def __init__(self, is_meta, is_quoted, is_raw, text):
        self.is_meta = is_meta
        self.is_quoted = is_quoted
        self.is_raw = is_raw
        self.text = text
