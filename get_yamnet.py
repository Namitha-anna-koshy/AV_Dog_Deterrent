import urllib.request
import tarfile
import os

# Official TF Hub download link
url = "https://tfhub.dev/google/yamnet/1?tf-hub-format=compressed"
filename = "yamnet.tar.gz"
extract_dir = "yamnet"

print("⏳ Disguising as a browser and downloading YAMNet...")

# The "Disguise" to bypass the 403 Forbidden error
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})

with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
    out_file.write(response.read())

print("📦 Extracting files...")
with tarfile.open(filename, "r:gz") as tar:
    tar.extractall(path=extract_dir)

# Clean up the zip file
os.remove(filename)

print("✅ SUCCESS! The 'yamnet' folder is ready for offline use.")