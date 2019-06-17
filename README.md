# ITCT_Decoder_HW

# JPEG :
TODO...

# Realtime MPEG-1 Decoder :
Decode I_ONLY, IP_ONLY, and IPB_ALL.M1V and dispaly image sequence via opencv highgui. (opencv 2.4.13.6).
    
![test image size](/MPEG_decoder/img/img.png)
## Enviroment :
- mingw32
- g++ 8.2.0
- opencv-2.4.13.6
## How to compile :
Using easpEMP to output image sequence :
```
g++ -IC:\\opencv\\build\\install\\include -LC:\\opencv\\build\\install\\x86\\mingw\\lib -g -o output.exe main.cpp decoder_utils.cpp ./easyBMP/EasyBMP.cpp -lopencv_core2413 -lopencv_highgui2413 -lopencv_imgproc2413
```
Only using opencv highgui to display :
```
g++ -IC:\\opencv\\build\\install\\include -LC:\\opencv\\build\\install\\x86\\mingw\\lib -g -o output.exe main.cpp decoder_utils.cpp -lopencv_core2413 -lopencv_highgui2413 -lopencv_imgproc2413
```
## How to execute :
```
output.exe <filename.M1V>
```
For example :
```
output.exe IPB_ALL.M1V
```
## Performance
- I_ONLY.M1V 3.8s in avg.
- IP_ONLY.M1V 3.6s in avg.
- IPB_ALL.M1V 3.2s in avg.
