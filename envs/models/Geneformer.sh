

home=$1
# ./scripts/models/Geneformer

mkdir -p $home/gits


cd $home/gits

git lfs install
git clone https://hf-mirror.com/ctheodoris/Geneformer # 全模型
# GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/ctheodoris/Geneformer
cd Geneformer

conda create -n Geneformer python=3.10 -y
conda activate Geneformer
pip install .


# 将pkl移动至dir
sitedir=$(python -c "import site; print(site.getsitepackages()[0])")/geneformer
cd geneformer
cp *.pkl $sitedir
cp gene_dictionaries_30m $sitedir -r


# 下载模型
mkdir -p ../models
cd ../models
# V1-10M
mkdir -p Geneformer-V1-10M
wget https://hf-mirror.com/ctheodoris/Geneformer/resolve/main/Geneformer-V1-10M/config.json?download=true -O Geneformer-V1-10M/config.json
wget https://hf-mirror.com/ctheodoris/Geneformer/resolve/main/Geneformer-V1-10M/model.safetensors?download=true -O Geneformer-V1-10M/model.safetensors
wget https://hf-mirror.com/ctheodoris/Geneformer/resolve/main/Geneformer-V1-10M/pytorch_model.bin?download=true -O Geneformer-V1-10M/pytorch_model.bin
wget https://hf-mirror.com/ctheodoris/Geneformer/resolve/main/Geneformer-V1-10M/training_args.bin?download=true -O Geneformer-V1-10M/training_args.bin

# V2-104M
mkdir -p Geneformer-V2-104M
wget https://hf-mirror.com/ctheodoris/Geneformer/resolve/main/Geneformer-V2-104M/config.json?download=true  -O Geneformer-V2-104M/config.json
wget https://hf-mirror.com/ctheodoris/Geneformer/resolve/main/Geneformer-V2-104M/model.safetensors?download=true  -O Geneformer-V2-104M/model.safetensors
wget https://hf-mirror.com/ctheodoris/Geneformer/resolve/main/Geneformer-V2-104M/pytorch_model.bin?download=true  -O Geneformer-V2-104M/pytorch_model.bin
wget https://hf-mirror.com/ctheodoris/Geneformer/resolve/main/Geneformer-V2-104M/training_args.bin?download=true  -O Geneformer-V2-104M/training_args.bin

# V2-104M_CLcancer
mkdir -p Geneformer-V2-104M_CLcancer
wget https://hf-mirror.com/ctheodoris/Geneformer/resolve/main/Geneformer-V2-104M_CLcancer/config.json?download=true -O  Geneformer-V2-104M_CLcancer/config.json
wget https://hf-mirror.com/ctheodoris/Geneformer/resolve/main/Geneformer-V2-104M_CLcancer/model.safetensors?download=true -O  Geneformer-V2-104M_CLcancer/model.safetensors
wget https://hf-mirror.com/ctheodoris/Geneformer/resolve/main/Geneformer-V2-104M_CLcancer/pytorch_model.bin?download=true -O  Geneformer-V2-104M_CLcancer/pytorch_model.bin
wget https://hf-mirror.com/ctheodoris/Geneformer/resolve/main/Geneformer-V2-104M_CLcancer/training_args.bin?download=true -O  Geneformer-V2-104M_CLcancer/training_args.bin

# V2-316M
mkdir -p Geneformer-V2-316M
wget https://hf-mirror.com/ctheodoris/Geneformer/resolve/main/Geneformer-V2-316M/config.json?download=true -O  Geneformer-V2-316M/config.json
wget https://hf-mirror.com/ctheodoris/Geneformer/resolve/main/Geneformer-V2-316M/model.safetensors?download=true -O  Geneformer-V2-316M/model.safetensors
wget https://hf-mirror.com/ctheodoris/Geneformer/resolve/main/Geneformer-V2-316M/pytorch_model.bin?download=true -O  Geneformer-V2-316M/pytorch_model.bin
wget https://hf-mirror.com/ctheodoris/Geneformer/resolve/main/Geneformer-V2-316M/training_args.bin?download=true -O  Geneformer-V2-316M/training_args.bin
