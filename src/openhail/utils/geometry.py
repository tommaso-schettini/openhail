import numpy as np
import scipy.spatial.distance as distances


def distance_matrix(a, b, squared=False):
    if squared:
        return distances.cdist(a, b, metric="sqeuclidean")
    return distances.cdist(a, b)


def get_distance(a, b, squared=False):
    if a.shape == b.shape:
        if a.shape == (2,):
            return element_distance(a, b, squared)
        return elementwise_distance(a, b, squared)
    return vector_distance(a, b, squared)


def vector_distance(a, b, squared=False):
    if len(b.shape) == 1:
        a, b = b, a

    if squared:
        return distances.cdist([a], b, metric="sqeuclidean").flatten()
    return distances.cdist([a], b).flatten()


def elementwise_distance(a, b, squared=False):
    if squared:
        return np.sum((a - b) ** 2, axis=1)
    return np.linalg.norm(a - b, axis=1)


def element_distance(a, b, squared=False):
    if squared:
        return np.sum((a - b) ** 2)
    return np.linalg.norm(a - b)


def cdist(o_coords, d_coords, matrix=False):
    """Computes the distance between o_coords and d_coords.

    If o_coords and d_coords have different shapes, then returns
    the distance matrix whose shape is (len(o_coords), len(d_coords)).
    (side note: returns a scalar if each is a single coordinate pair)

    If o_coords and d_coords have the same shape, then returns the
    element-wise distance between members in o_coords and d_coords.

    If o_coords and d_coords have the same shape and you want the distance
    matrix, then you must specify matrix=True.
    """

    # just want distance between two (non-complex) points
    if o_coords.shape == d_coords.shape == (2,):
        return np.abs(complex(*o_coords) - complex(*d_coords))

    # want the distance matrix
    elif o_coords.shape != d_coords.shape or matrix:
        result = np.abs(to_complex_array(o_coords).T - to_complex_array(d_coords))
        # return vector if result is 1xN or Nx1
        return result.flatten() if 1 in result.shape else result

    # want element-wise distances
    else:
        result = np.abs(
            to_complex_array(o_coords) - to_complex_array(d_coords)
        ).flatten()
        # if length is one, return scalar (only happens if passed two complex
        # singletons; unlikely but possible)
        return result[0] if result.size == 1 else result


def to_complex_array(coords):
    """
    Input coords should have shape (N,2); 2 being x and y (or just (2,) if just one pair
    of coords).
    """

    # promote to ndarray if not already
    is_np_array = isinstance(coords, np.ndarray)
    a = np.array(coords) if not is_np_array else coords

    # see if it's already a properly formatted complex array
    if a.dtype == complex and a.shape == (1, a.size):
        return a

    expected_matrix_shape = (a.size // 2, 2)
    if a.shape not in (expected_matrix_shape, (2,)):
        raise ValueError(
            "Improper dimensions for converting coordinates to complex: "
            f"got {a.shape}, expected (N, 2) or (2,)."
        )

    # if it's just one coordinate pair, add first dimension
    if a.shape == (2,):
        a = a[None, :]

    # make complex array
    result = np.empty((1, len(a)), dtype=complex)
    result.real = a[:, 0]
    result.imag = a[:, 1]

    return result
