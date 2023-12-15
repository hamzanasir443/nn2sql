import requests

def download_file(url, local_filename):
    """
    Downloads a file from a given URL and saves it locally.
    """
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
    return local_filename

# URL of the file to be downloaded (Direct Download Link)
file_url = "https://www.dropbox.com/scl/fi/5t8lhecp8fx8493fjx90n/mnist_train.csv?rlkey=xewnisrqylpznrfr0ekfu32qt&dl=1"
local_file_name = "mnist_train.csv"

# Download the file
download_file(file_url, local_file_name)
print(f"Downloaded '{local_file_name}' successfully.")

