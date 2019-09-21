import pandas as pd


def read_10x_hdf5(file_name, sparse=False, **kwargs):
    """
    Load a HDF5 file using scanpy
    Process it into a DataFrame
    :param file_name: str
    :param sparse: bool
        Use sparse datatypes for the array
    :return data_frame: pd.DataFrame [N x G]
    """

    import scanpy
    ann_data = scanpy.read_10x_h5(file_name)
    if sparse:
        data_frame = pd.DataFrame.sparse.from_spmatrix(ann_data.X,
                                                       index=ann_data.obs_names,
                                                       columns=ann_data.var_names)
    else:
        data_frame = pd.DataFrame(ann_data.X.todense(),
                                  index=ann_data.obs_names,
                                  columns=ann_data.var_names)
    return data_frame
