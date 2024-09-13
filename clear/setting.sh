apt-get update -y
apt-get install -y libgl1-mesa-glx
apt-get install -y libglib2.0-0
apt-get install wget
wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000307/data/code.tar.gz
wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000307/data/data.tar.gz
tar -zxvf data.tar.gz
tar --strip-components=1 -xzf code.tar.gz
pip install -r requirements.txt