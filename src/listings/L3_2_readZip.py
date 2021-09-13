""" Get data from MS-Excel files, which are stored zipped on the WWW """

# author: Thomas Haslwanter, date: Sept-2021

# Import standard packages
import pandas as pd

# additional packages
import io
from zipfile import ZipFile
from urllib.request import urlopen
from typing import List

    
def get_dataDobson(url: str, in_file: str) -> pd.DataFrame:
    """ Extract data from a zipped-archive on the web.
    
    Parameters
    ----------
    url : URL of the archived zip-file
    
    Returns
    -------
    zipdata : bytestream/buffer with the zipped archive
    """

    # get the zip-archive
    archive = urlopen(url).read()

    # make the archive available as a byte-stream
    zipdata = io.BytesIO()
    zipdata.write(archive)

    # extract the requested file from the archive, as a pandas XLS-file
    myzipfile = ZipFile(zipdata)
    xlsfile = myzipfile.open(in_file)

    # read the xls-file into Python, using Pandas, and return the extracted data
    xls = pd.ExcelFile(xlsfile)
    df  = xls.parse('Sheet1', skiprows=2)

    return df


if __name__ == '__main__':
    # Select archive (on the web) and the file in the archive
    url = 'https://work.thaslwanter.at/sapy/GLM.dobson.data.zip'
    in_file = 'Table 2.8 Waist loss.xls'

    df = get_dataDobson(url, in_file)
    print(df)

    input('All done!')
