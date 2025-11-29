
home=$1

mkdir -p $home
cd $home

conda create -n scGPT python=3.9 r-base -y
conda activate scGPT
pip install scanpy -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
# https://download.pytorch.org/whl/cu117
# cd /data/run01/sczd231/disk/Projects/BenchMark/Methods/scGPT/git/scGPT-main
# pip install .
pip install scgpt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install -U scvi-tools[cuda]==0.16.0 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install torchtext==0.15.2 --index-url  https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --force
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install wandb -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install --upgrade 'numpy<2'
pip install hdf5plugin -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install anndata==0.10.8 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 下载模型
mkdir -p ../models
cd ../models

mkdir -p scGPT_human
wget https://drive.google.com/file/d/1hh2zGKyWAx3DyovD30GStZ3QlzmSqdk1/view?usp=drive_link -O scGPT_human/args.json
wget https://drive.google.com/file/d/14AebJfGOUF047Eg40hk57HCtrb0fyDTm/view?usp=drive_link -O scGPT_human/best_model.pt
wget https://drive.google.com/file/d/1H3E_MJ-Dl36AQV6jLbna2EdvgPaqvqcC/view?usp=drive_link -O scGPT_human/vocab.json

mkdir -p scGPT_CP
wget https://drive.google.com/file/d/1H3E_MJ-Dl36AQV6jLbna2EdvgPaqvqcC/view?usp=drive_link -O scGPT_CP/args.json
wget https://drive.google.com/file/d/1x1SfmFdI-zcocmqWAd7ZTC9CTEAVfKZq/view?usp=drive_link -O scGPT_CP/best_model.pt
wget https://drive.google.com/file/d/1jfT_T5n8WNbO9QZcLWObLdRG8lYFKH-Q/view?usp=drive_link -O scGPT_CP/vocab.json

