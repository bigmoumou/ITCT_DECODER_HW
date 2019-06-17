#include <iostream>
#include <fstream>
#include <stdio.h>
#include "decoder_utils.h"

#include <time.h>

int main(int argc, char **argv)
{
    clock_t tStart = clock();
    
    // I_ONLY.M1V,  IP_ONLY.M1V,  IPB_ALL.M1V
    string filename = argv[1];
    Decoder processor;
    processor.read_data(filename);
    processor.video_sequence();
    
    printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
    return 0;
}