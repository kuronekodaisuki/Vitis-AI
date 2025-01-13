# Unzip wide_resnet50_2 quantized model
Unzip quantized.zip

# Compile for V70
vai_c_xir \
 -x wuantized.xmodel\
 -a /opt/vitis_ai/compiler/arch/DPUCV2DX8G/V70/arch.json\
 -n wide_resnet50_2
 -o .