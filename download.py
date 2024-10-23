import requests

url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
local_filename = 'sam_vit_h_4b8939.pth'

with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192): 
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                print(f"Downloaded {f.tell() // (1024 * 1024)} MB", end='\r')

print("Download completed!")
