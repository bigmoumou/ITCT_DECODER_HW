#include <iostream>
#include <fstream>
#include <stdio.h>
#include "decoder_utils.h"

#include <time.h>

int main(int argc, char **argv)
{
    clock_t tStart = clock();

    Decoder processor;
//    string filename = "C:\\Users\\o1r2g\\OneDrive\\Desktop\\cpp_tutorial\\CPPWorkspace\\MPEG1_decoder\\I_ONLY.M1V";
//    string filename = "C:\\Users\\o1r2g\\OneDrive\\Desktop\\cpp_tutorial\\CPPWorkspace\\MPEG1_decoder\\IP_ONLY.M1V";
    string filename = "C:\\Users\\o1r2g\\OneDrive\\Desktop\\cpp_tutorial\\CPPWorkspace\\MPEG1_decoder\\IPB_ALL.M1V";
    processor.read_data(filename);
    processor.video_sequence();

    printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
    return 0;
}