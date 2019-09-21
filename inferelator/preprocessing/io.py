import pandas as pd
from inferelator import utils


def read_10x_hdf5(file_name, sparse=False, **kwargs):
    """
    Load a HDF5 file using scanpy
    Process it into a DataFrame
    :param file_name: str
    :param sparse: bool
        Use sparse datatypes for the array
    :return data_frame: pd.DataFrame [N x G]
    """

    # Import scanpy
    try:
        import scanpy
    except ImportError as err:
        raise ImportError("Install scanpy to load hdf5 files") from err

    ann_data = scanpy.read_10x_h5(file_name)
    utils.Debug.vprint("{f} loaded into {sh} annotated data frame".format(f=file_name, sh=ann_data.X.shape))

    data_frame = pd.DataFrame.sparse.from_spmatrix(ann_data.X, index=ann_data.obs_names, columns=ann_data.var_names)

    if sparse:
        utils.Debug.vprint("Data processed into {sh} sparse DataFrame".format(sh=data_frame.shape))
        return data_frame
    else:
        utils.Debug.vprint("Data processed into {sh} dense DataFrame".format(sh=data_frame.shape))
        return data_frame.sparse.to_dense()
