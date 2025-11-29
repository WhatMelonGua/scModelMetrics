
# app.conda forge

conda create -n scModelMetrics python=3.10 bioconda::snakemake -y
conda activate scModelMetrics

pip install --upgrade pulp==2.7.0
pip install scib -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install scanpy -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install igraph -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install leidenalg -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install hdf5plugin -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install scikit-learn -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install scikit-misc -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

pip install huggingface_hub
# 国内hugging-face镜像
export HF_ENDPOINT=https://hf-mirror.com