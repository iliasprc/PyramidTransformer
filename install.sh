git clone https://github.com/iliasprc/PyramidTransformer.git
cd PyramidTransformer
virtualenv -p python3 pyramidtransformer
source pyramidtransformer/bin/activate
pip install -r requirements.txt

git clone https://github.com/linziyi96/pytorch.git
cd pytorch
git submodule update --init --recursive
git checkout 3d-depthwise
python setup.py install
cd ..
git clone https://github.com/pytorch/vision.git
git checkout tags/v0.6.0
cd vision
python setup.py install

