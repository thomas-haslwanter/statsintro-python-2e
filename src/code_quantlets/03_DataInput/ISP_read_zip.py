"""Get data from MS-Excel files, which are stored zipped on the WWW. """

# author:   Thomas Haslwanter
# date:     Aug-2021

# Import standard packages
import pandas as pd

# additional packages
import io
import zipfile
from urllib.request import urlopen
    

def get_data_dobson(url: str, inFile: str) -> pd.DataFrame:
    """ Extract data from a zipped-archive on the web. """

    # get the zip-archive
    GLM_archive = urlopen(url).read()

    # make the archive available as a byte-stream
    zip_data = io.BytesIO()
    zip_data.write(GLM_archive)

    # extract the requested file from the archive, as a pandas XLS-file
    my_zipfile = zipfile.ZipFile(zip_data)
    xls_file = my_zipfile.open(inFile)

    # read the xls-file into Python, using Pandas, and return the extracted data
    xls = pd.ExcelFile(xls_file)
    df  = xls.parse('Sheet1', skiprows=2)

    return df


if __name__ == '__main__':
    # Select archive (on the web) and the file in the archive
    # url = 'https://www.routledge.com/downloads/K32369/GLM.dobson.data.zip'
    url = 'https://work.thaslwanter.at/sapy/GLM.dobson.data.zip'
    inFile = r'Table 2.8 Waist loss.xls'

    df = get_data_dobson(url, inFile)
    print(df)

    #input('All done!')
