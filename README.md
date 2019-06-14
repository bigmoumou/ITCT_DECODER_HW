# ITCT_Decoder_HW

# JPEG

# MPEG-1 Decoder
Decode I_ONLY, IP_ONLY, and IPB_ALL.M1V from videos to bmp images. (output by easyBMP)    
Dispaly images sequence via opencv highgui. (opencv 2.4.13.6)
```
g++ -IC:\\opencv\\build\\install\\include -LC:\\opencv\\build\\install\\x86\\mingw\\lib -g -o output.exe main.cpp decoder_utils.cpp ./easyBMP/EasyBMP.cpp -lopencv_core2413 -lopencv_highgui2413 -lopencv_imgproc2413
mpeg.exe > out.txt
```
