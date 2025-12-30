import numpy as np

def split_vectors(vectors, split_num, start_index=0):
    """
    Split the vectors into two parts based on the given split ratio.
    
    :param vectors: numpy array of vectors to be split
    :param split_ratio: ratio to split the vectors (default is 0.8)
    :return: two numpy arrays containing the split vectors
    """

    split_size = split_num
    if start_index + split_size > len(vectors):
        raise ValueError("Start index and split size exceed the length of the vectors.")

    part = vectors[start_index:start_index + split_size]
    end_index = start_index + split_size

    return part, end_index