import black
from collections import deque


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


def read_last_n_lines(filename: str, n: int) -> str:
    """Read the last n lines of a file efficiently.

    Args:
        filename: Path to the file to read
        n: Number of lines to read from end

    Returns:
        String containing the last n lines
    """

    # Use deque with maxlen to efficiently store only n lines
    lines = deque(maxlen=n)

    # Read file in chunks from end
    with open(filename, "rb") as f:
        # Seek to end of file
        f.seek(0, 2)
        file_size = f.tell()

        # Start from end, read in 8KB chunks
        chunk_size = 8192
        position = file_size

        # Read chunks until we have n lines or reach start
        while position > 0 and len(lines) < n:
            # Move back one chunk
            chunk_size = min(chunk_size, position)
            position -= chunk_size
            f.seek(position)
            chunk = f.read(chunk_size).decode()

            # Split into lines and add to deque
            lines.extendleft(chunk.splitlines())

        # If we're not at file start, we may have a partial first line
        if position > 0:
            # Read one more chunk to get complete first line
            position -= 1
            f.seek(position)
            chunk = f.read(chunk_size).decode()
            lines[0] = chunk[chunk.rindex("\n") + 1 :] + lines[0]

    return "\n".join(lines)
