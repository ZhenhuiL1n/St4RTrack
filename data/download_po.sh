# # Download point_odyssey
# mkdir -p point_odyssey
# cd point_odyssey
# # train
# gdown --id 1ivaHRZV6iwxxH4qk8IAIyrOF9jrppDIP
# # test
# gdown --id 1jn8l28BBNw9f9wYFmd5WOCERH48-GsgB
# # sample
# gdown --id 1dnl9XMImdwKX2KcZCTuVDhcy5h8qzQIO
# # unzip all *.tar.gz
# find . -name "*.tar.gz" -exec tar -zxvf {} \;
# # remove all zip files
# find . -name "*.tar.gz" -exec rm {} \;


# Set your target directory
DATA_DIR="/home/longnhat/Lin_workspace/8TB2/Lin/nas-train/point_odyssey"

# Create directory if not exists
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "Downloading files to: $DATA_DIR"

# train
gdown --id 1ivaHRZV6iwxxH4qk8IAIyrOF9jrppDIP

# test
gdown --id 1jn8l28BBNw9f9wYFmd5WOCERH48-GsgB

# sample
gdown --id 1dnl9XMImdwKX2KcZCTuVDhcy5h8qzQIO

# unzip all *.tar.gz
find "$DATA_DIR" -name "*.tar.gz" -exec tar -zxvf {} \;

# remove all tar.gz files
find "$DATA_DIR" -name "*.tar.gz" -exec rm {} \;

echo "All downloads and extractions completed!"
