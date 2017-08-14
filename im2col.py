import numpy as np


def get_im2col_indices(x_shape, field_height=3, field_width=3, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) / stride + 1
    out_width = (W + 2 * padding - field_width) / stride + 1

    i0 = np.repeat(np.arange(field_height, dtype='int32'), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height, dtype='int32'), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width, dtype='int32'), int(out_height))
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C, dtype='int32'), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height=3, field_width=3, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    if padding > 0:
        # Zero-pad the input
        p = padding
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    else:
        x_padded = x

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    # (view, batch x input_channel)
    cols = cols.transpose(1, 0, 2).reshape(field_height * field_width * C, -1)
    return cols


# cols = (27, 60) = (in_c * in_h * in_w, out_h * out_w * batch)
def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype) # (10, 3, 4, 5)

    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N) # (27, 6, 10)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1) # (10, 27, 6) (batch, in_c * in_h * in_w, out_h * out_w)

    # (27, 1), (27, 6), (27, 6) input to output binding
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    # (10, 3, 4, 5) + (10, 27, 6)
    x_padded[:, k, i, j] = cols_reshaped

    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]
