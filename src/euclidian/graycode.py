


def gray(nbits):
    """Generate a gray code as a tuple of 0 and 1 integers."""
    code = [0] * nbits
    yield tuple(code)

    for term in range(2, (1 << nbits) + 1):
        if term % 2 != 0:
            for i in range(-1, -nbits, -1):
                if code[i] == 1:
                    code[i - 1] ^= 1
                    break
        else:
            code[-1] ^= 1

        yield tuple(code)

def signed_gray(nbits):
    """Generate a gray code as a tupe of -1 and +1 integers."""
    for code in gray(nbits):
        yield tuple(-1 + 2*c for c in code)

if __name__ == '__main__':
    print(list(gray(4)))
