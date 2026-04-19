# a small tool function,
#   calculate the length of sample_x and sample_y,
#   they should be equal


def size(sample_x, sample_y):
    if len(sample_x) == len(sample_y):
        length = len(sample_y)
    else:
        print("Number of samples x and y do not match!")
        return None
    return length
