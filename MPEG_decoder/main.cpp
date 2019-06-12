#include <iostream>
#include <fstream>
#include <stdio.h>
#include "decoder_utils.h"
int main(int argc, char **argv)
{
	Decoder processor;
    string filename = "C:\\Users\\o1r2g\\OneDrive\\Desktop\\cpp_tutorial\\CPPWorkspace\\MPEG1_decoder\\IPB_ALL.M1V";
    processor.read_data(filename);
    processor.video_sequence();
	return 0;
}
