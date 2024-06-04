import urllib.request
import tarfile

def download_dataset(url: str, filename: str):
    urllib.request.urlretrieve(url, filename)

def extract_dataset(tar_filename: str):
    emails = []
    labels = []
    with tarfile.open(tar_filename, "r:gz") as tar:
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f is not None:
                content = f.read()
                if 'enron1/ham' in member.name:
                    emails.append(content.decode('utf-8', errors='ignore'))
                    labels.append('ham')
                elif 'enron1/spam' in member.name:
                    emails.append(content.decode('utf-8', errors='ignore'))
                    labels.append('spam')
    return emails, labels

