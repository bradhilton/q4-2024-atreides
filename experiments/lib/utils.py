import black
from collections import deque
import time
import torch
from typing import Any, Callable, Optional, ParamSpec, Sequence, TypeVar, Union


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


P = ParamSpec("P")
T = TypeVar("T")


def return_exception(callable: Callable[P, T]) -> Callable[P, Union[T, BaseException]]:
    """Decorator to return exception instead of raising it.

    Args:
        callable: Function to decorate

    Returns:
        Decorated function that returns exception instead of raising it
    """

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Union[T, BaseException]:
        try:
            return callable(*args, **kwargs)
        except BaseException as exception:
            return exception

    return wrapper


class Timer:
    def __init__(self, description: str):
        self.description = description
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        if not exc_type:
            seconds = time.time() - self.start_time
            print(f"{self.description} in {seconds:.2f}s ✓")


def truncate_pad(
    input: torch.Tensor,
    shape: Sequence[int],
    mode: str = "constant",
    value: Optional[float] = None,
) -> torch.Tensor:
    """Truncates or pads a tensor to match the target shape.

    For each dimension i, if shape[i] is:
    - -1: Leave that dimension unchanged
    - < input.shape[i]: Truncate to first shape[i] elements
    - > input.shape[i]: Pad with value to reach shape[i] elements

    Args:
        input: Input tensor to reshape
        shape: Target shape, with -1 indicating unchanged dimensions
        mode: Padding mode to pass to torch.nn.functional.pad
        value: Pad value to pass to torch.nn.functional.pad

    Returns:
        Tensor with dimensions matching shape (except where -1)
    """
    result = input
    for i in range(len(shape)):
        if shape[i] == -1:
            continue
        if shape[i] < input.shape[i]:
            # Truncate on this dimension
            slicing = [slice(None)] * len(input.shape)
            slicing[i] = slice(0, shape[i])
            result = result[tuple(slicing)]
        elif shape[i] > input.shape[i]:
            # Start of Selection
            padding = [0] * (2 * len(input.shape))
            padding[2 * (len(input.shape) - i - 1) + 1] = shape[i] - input.shape[i]
            result = torch.nn.functional.pad(result, padding, mode=mode, value=value)
    return result
