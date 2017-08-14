import numpy as np


def get_im2col_indices(x_shape, f_h=3, f_w=3, padding=1, stride=1):
    # First figure out what the size of the output should be
    batch, in_c, in_h, in_w = x_shape
    assert (in_h + 2 * padding - f_h) % stride == 0
    assert (in_w + 2 * padding - f_h) % stride == 0

    o_h = (in_h + 2 * padding - f_h) / stride + 1
    o_w = (in_w + 2 * padding - f_w) / stride + 1

    i0 = np.repeat(np.arange(f_h, dtype='int32'), f_w)
    i0 = np.tile(i0, in_c)
    i1 = stride * np.repeat(np.arange(o_h, dtype='int32'), o_w)
    j0 = np.tile(np.arange(f_w), f_h * in_c)
    j1 = stride * np.tile(np.arange(o_w, dtype='int32'), int(o_h))
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(in_c, dtype=np.int32), f_h * f_w).reshape(-1, 1)

    # indexes bindings by (in_c, in_h, in_w)
    return k, i, j


def im2col_indices(x, f_h=3, f_w=3, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    if padding > 0:
        # Zero-pad the input
        p = padding
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    else:
        x_padded = x

    in_c = x.shape[1]

    # (27, 1), (27, 6), (27, 6) input to output binding
    k, i, j = get_im2col_indices(x.shape, f_h, f_w, padding, stride)

    # (10, 27, 6) or (batch, in_c * f_h * f_w, o_h * o_w)
    cols = x_padded[:, k, i, j]

    # (27, 60) or (view, batch * o_h * o_w) view = in_c * f_h * f_w
    cols = cols.transpose(1, 0, 2).reshape(f_h * f_w * in_c, -1)
    return cols


# cols = (27, 60) or (in_c * in_h * in_w, out_h * out_w * batch)
def col2im_indices(cols, x_shape, f_h=3, f_w=3, padding=1, stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    batch, in_c, in_h, in_w = x_shape

    if padding > 0:
        h_padded, w_padded = in_h + 2 * padding, in_w + 2 * padding
        # (10, 3, 4, 5) with 0 padding
        x_padded = np.zeros((batch, in_c, h_padded, w_padded), dtype=cols.dtype)
    else:
        x_padded = np.zeros(x_shape, dtype=cols.dtype)

    # (27, 6, 10) or (in_c * in_h * in_w, out_h * out_w, batch)
    cols_reshaped = cols.reshape(in_c * f_h * f_w, -1, batch)

    # (10, 27, 6) or (batch, in_c * in_h * in_w, out_h * out_w)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)

    # (27, 1), (27, 6), (27, 6) input to output binding
    k, i, j = get_im2col_indices(x_shape, f_h, f_w, padding, stride)

    # (10, 3, 4, 5) += (10, 27, 6)
    x_padded[:, k, i, j] = cols_reshaped

    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]
