#!/bin/bash

# Build libjpeg
cd ../2_libraries/libjpeg/
./configure --enable-shared --enable-static --prefix=$(pwd)/../_installdir/libjpeg --exec-prefix=$(pwd)/../_installdir/libjpeg
cd ../../0_preparatory_operations
make --directory=../2_libraries/libjpeg/ -j $(nproc)
make --directory=../2_libraries/libjpeg/ test

# Build libjpeg-turbo
cmake -S ../2_libraries/libjpeg-turbo -B ../2_libraries/libjpeg-turbo/build -DCMAKE_INSTALL_PREFIX=$(pwd)/../2_libraries/_installdir/libjpeg-turbo -DCMAKE_BUILD_TYPE="Release"
cmake --build ../2_libraries/libjpeg-turbo/build --config Release --parallel

# Build jpegli
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -S ../2_libraries/jpegli/ -B ../2_libraries/jpegli/build -DJPEGXL_INSTALL_JPEGLI_LIBJPEG=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_INSTALL_PREFIX=$(pwd)/../2_libraries/_installdir/jpegli
cmake --build ../2_libraries/jpegli/build --config Release --parallel

# Build libjxl
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -S ../2_libraries/libjxl/ -B ../2_libraries/libjxl/build -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_INSTALL_PREFIX=$(pwd)/../2_libraries/_installdir/libjxl
cmake --build ../2_libraries/libjxl/build --config Release --parallel

# Build gpujpeg
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native -B ../2_libraries/gpujpeg/build -S ../2_libraries/gpujpeg -DHUFFMAN_GPU_CONST_TABLE=ON -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_INSTALL_PREFIX=$(pwd)/../2_libraries/_installdir/gpujpeg
cmake --build ../2_libraries/gpujpeg/build --config Release --parallel

# Build libwebp
cmake -S ../2_libraries/libwebp -B ../2_libraries/libwebp/build -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_INSTALL_PREFIX=$(pwd)/../2_libraries/_installdir/libwebp
cmake --build ../2_libraries/libwebp/build --config Release --parallel

# Build libavif
cmake -DCMAKE_BUILD_TYPE=Release -S ../2_libraries/libavif -B ../2_libraries/libavif/build -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=OFF -DAVIF_CODEC_AOM=LOCAL -DAVIF_LIBYUV=LOCAL -DAVIF_LIBSHARPYUV=LOCAL -DAVIF_JPEG=LOCAL -DAVIF_ZLIBPNG=LOCAL -DAVIF_CODEC_DAV1D=LOCAL -DAVIF_CODEC_LIBGAV1=LOCAL -DAVIF_CODEC_RAV1E=LOCAL -DAVIF_CODEC_SVT=LOCAL -DAVIF_BUILD_APPS=ON -DCMAKE_INSTALL_PREFIX=$(pwd)/../2_libraries/_installdir/libavif
cmake --build ../2_libraries/libavif/build --config Release --parallel # RAM HUNGRY, REMOVE --parallel IF IT CRASHES
  
# Build libavif for AV2 
cmake -DCMAKE_BUILD_TYPE=Release -S ../2_libraries/libavif-av2 -B ../2_libraries/libavif-av2/build -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=OFF -DAVIF_CODEC_AVM=LOCAL -DAVIF_LIBYUV=LOCAL -DAVIF_LIBSHARPYUV=LOCAL -DAVIF_JPEG=LOCAL -DAVIF_ZLIBPNG=LOCAL -DAVIF_BUILD_APPS=ON -DCMAKE_INSTALL_PREFIX=$(pwd)/../2_libraries/_installdir/libavif-av2
cmake --build ../2_libraries/libavif-av2/build --config Release --parallel # RAM HUNGRY, REMOVE --parallel IF IT CRASHES

# Build libx264
cd ../2_libraries/x264/
./configure --enable-lto --enable-pic --enable-shared --enable-static --prefix=$(pwd)/../_installdir/x264 --exec-prefix=$(pwd)/../_installdir/x264 # debug and lto are mutually exclusive, when deploying remove debug
cd ../../0_preparatory_operations
make --directory=../2_libraries/x264/ -j $(nproc)

# Build VC-2 reference
patch ../2_libraries/vc2-reference/configure.ac ./patches/vc2-reference_configure.ac_patch
cd ../2_libraries/vc2-reference/
# configure.ac needs to be modified in the following way:
# add AX_CXX_COMPILE_STDCXX([11], [noext], [mandatory]) below AC_PROG_CXX 
# add AC_PROG_CC below AC_PROG_CXX 
# fix VC2REFERENCE_LIBS with VC2REFERENCE_LIBS="\$(top_builddir)/src/Library/libVC2-$VC2REFERENCE_MAJORMINOR.la"
# fix AS_NANO with AS_NANO([VC2REFERENCE_CVS=no], [VC2REFERENCE_CVS=yes])
./autogen.sh
./configure --enable-shared --enable-static --prefix=$(pwd)/../_installdir/vc2-reference --exec-prefix=$(pwd)/../_installdir/vc2-reference
cd ../../0_preparatory_operations
make --directory=../2_libraries/vc2-reference/ -j $(nproc)

# Build VC-2 HQ
echo sed '#!/bin/bash' > ../2_libraries/vc2-hq/encode/duplicate-transform
echo sed '"s/$3/$4/g" "$1" > "$2"' >> ../2_libraries/vc2-hq/encode/duplicate-transform
patch ../2_libraries/vc2-hq/encode/configure.ac ./patches/vc2-hq-encode_configure.ac_patch
patch ../2_libraries/vc2-hq/decode/configure.ac ./patches/vc2-hq-decode_configure.ac_patch
cd ../2_libraries/vc2-hq/encode
# configure.ac needs to be modified in the following way:
# move AX_CXX_COMPILE_STDCXX_11 below AC_PROG_CXX 
# add AC_PROG_CC below AC_PROG_CXX 
# fix VC2REFERENCE_LIBS with VC2REFERENCE_LIBS="\$(top_builddir)/src/Library/libVC2-$VC2REFERENCE_MAJORMINOR.la"
# fix AS_NANO with AS_NANO([VC2HQENCODE_CVS=no],[VC2HQENCODE_CVS=yes])
./autogen.sh
./configure --enable-shared --enable-static --prefix=$(pwd)/../../_installdir/vc2-hq/encode --exec-prefix=$(pwd)/../../_installdir/vc2-hq/encode
cd ../decode
# configure.ac needs to be modified in the following way:
# move AX_CXX_COMPILE_STDCXX_11 below AC_PROG_CXX 
# add AC_PROG_CC below AC_PROG_CXX 
# fix VC2REFERENCE_LIBS with VC2REFERENCE_LIBS="\$(top_builddir)/src/Library/libVC2-$VC2REFERENCE_MAJORMINOR.la"
# fix AS_NANO with AS_NANO([VC2HQDECODE_CVS=no],[VC2HQDECODE_CVS=yes])
./autogen.sh
./configure --enable-shared --enable-static --prefix=$(pwd)/../../_installdir/vc2-hq/decode --exec-prefix=$(pwd)/../../_installdir/vc2-hq/decode
cd ../../../0_preparatory_operations
make --directory=../2_libraries/vc2-hq/encode/ -j $(nproc)
make --directory=../2_libraries/vc2-hq/decode/ -j $(nproc)
