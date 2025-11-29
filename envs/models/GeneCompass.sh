
home=$1

mkdir -p $home/gits
cd $home/gits
git clone https://github.com/xCompass-AI/GeneCompass.git

conda create -n GeneCompass python=3.10 -y
conda activate GeneCompass
cd GeneCompass
# MUST run with git folder
# export PATH="$(pwd):$PATH"
pip install -r requirements.txt
