from typing import Generator, Tuple


def generate_upper_triangle(
    lower_bound: int, upper_bound: int, step_size: int = 1
) -> Generator(Tuple[int, int]):
    # generator
    for i in range(lower_bound, upper_bound, step_size):
        for j in range(i+step_size, upper_bound+1, step_size):
            yield i, j


def generate_off_diagonal(
    lower_bound: int, upper_bound: int, step_size: int = 1
) -> Generator(Tuple[int, int]):
    # generator
    for i in range(lower_bound, upper_bound, step_size):
        yield i, i+step_size


STEP_GENERATOR = {
    "full_triangle": generate_upper_triangle,
    "off_diagonal": generate_off_diagonal
}
