# include "../EasyBMP.h"
using namespace std;


int main()
{
	BMP AnImage;
	AnImage.SetSize (640,480);
	AnImage.SetBitDepth (8);
	AnImage (14,18) -> Red =0;
	AnImage (14,18) -> Green =0;
	AnImage (14,18) -> Blue =0;
	AnImage (14,18) -> Alpha =0;
	AnImage.WriteToFile ("output.bmp");

return 0;
}