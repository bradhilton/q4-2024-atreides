import black


def black_print(
    value: object,
) -> None:
    """
    Prints the value with black formatting.

    Args:
        value (object): The value to print.

    Note:
        The string representation of the value must be valid Python code.
    """
    print(
        black.format_str(str(value), mode=black.Mode()).strip(),
    )
