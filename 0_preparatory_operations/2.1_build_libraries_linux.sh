#!/bin/bash

# Build libjpeg
cd ../2_libraries/libjpeg/
./configure
cd ../../0_preparatory_operations
make --directory=../2_libraries/libjpeg/ -j $(nproc)
make --directory=../2_libraries/libjpeg/ test

# Build libjpeg-turbo
cmake -S ../2_libraries/libjpeg-turbo -B ../2_libraries/libjpeg-turbo/build -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build ../2_libraries/libjpeg-turbo/build --parallel

# Build libjxl + jpegli
cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=OFF -S ../2_libraries/libjxl-jpegli/ -B ../2_libraries/libjxl-jpegli/build -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ 
cmake --build ../2_libraries/libjxl-jpegli/build --config Debug --parallel

# Build gpujpeg
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES=native -B ../2_libraries/gpujpeg/build -S ../2_libraries/gpujpeg -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build ../2_libraries/gpujpeg/build --config Debug --parallel

# Build libwebp
cmake -S ../2_libraries/libwebp -B ../2_libraries/libwebp/build -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build ../2_libraries/libwebp/build --config Debug --parallel

# Build libavif
cmake -DCMAKE_BUILD_TYPE=Debug -S ../2_libraries/libavif -B ../2_libraries/libavif/build -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=OFF -DAVIF_CODEC_AOM=LOCAL -DAVIF_LIBYUV=LOCAL -DAVIF_LIBSHARPYUV=LOCAL -DAVIF_JPEG=LOCAL -DAVIF_ZLIBPNG=LOCAL -DAVIF_CODEC_DAV1D=LOCAL -DAVIF_CODEC_LIBGAV1=LOCAL -DAVIF_CODEC_RAV1E=LOCAL -DAVIF_CODEC_SVT=LOCAL -DAVIF_BUILD_APPS=ON 
scmake --build ../2_libraries/libavif/build --config Debug --parallel # RAM HUNGRY, REMOVE --parallel IF IT CRASHES
  
# Build libavif for AV2 
cmake -DCMAKE_BUILD_TYPE=Debug -S ../2_libraries/libavif-av2 -B ../2_libraries/libavif-av2/build -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=OFF -DAVIF_CODEC_AVM=LOCAL -DAVIF_LIBYUV=LOCAL -DAVIF_LIBSHARPYUV=LOCAL -DAVIF_JPEG=LOCAL -DAVIF_ZLIBPNG=LOCAL -DAVIF_BUILD_APPS=ON 
cmake --build ../2_libraries/libavif-av2/build --config Debug --parallel # RAM HUNGRY, REMOVE --parallel IF IT CRASHES

# Build libx264
cd ../2_libraries/x264/
./configure --enable-lto --enable-pic --enable-shared --enable-static --enable-debug # debug and lto are mutually exclusive, when deploying remove debug
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
./configure --enable-shared --enable-static
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
./configure --enable-shared --enable-static
cd ../decode
# configure.ac needs to be modified in the following way:
# move AX_CXX_COMPILE_STDCXX_11 below AC_PROG_CXX 
# add AC_PROG_CC below AC_PROG_CXX 
# fix VC2REFERENCE_LIBS with VC2REFERENCE_LIBS="\$(top_builddir)/src/Library/libVC2-$VC2REFERENCE_MAJORMINOR.la"
# fix AS_NANO with AS_NANO([VC2HQDECODE_CVS=no],[VC2HQDECODE_CVS=yes])
./autogen.sh
./configure --enable-shared --enable-static
cd ../../../0_preparatory_operations
make --directory=../2_libraries/vc2-hq/encode/ -j $(nproc)
make --directory=../2_libraries/vc2-hq/decode/ -j $(nproc)
