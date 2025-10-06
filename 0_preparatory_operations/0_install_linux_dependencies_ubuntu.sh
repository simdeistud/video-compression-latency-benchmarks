sudo apt-get install -y git build-essential cmake automake pkg-config ffmpeg yasm nasm meson ninja-build rustup libbrotli-dev libgif-dev libjpeg-dev libopenexr-dev libpng-dev libwebp-dev clang libglew-dev freeglut3-dev libglfw3-dev libglfw3 libtool libboost-all-dev

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit nvidia-cuda-toolkit nvidia-gds

rustup update beta

wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-noble.list https://packages.lunarg.com/vulkan/lunarg-vulkan-noble.list
sudo apt-get update
sudo apt-get install -y vulkan-sdk

#git clone https://github.com/intel/libvpl
#cd libvpl
#sudo script/bootstrap
#cmake -B _build -DCMAKE_INSTALL_PREFIX=/usr/local/intelvpl
#cmake --build _build
#cmake --install _build
