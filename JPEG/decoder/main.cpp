#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <bitset>

#include "decode_utils.h"
using namespace std;


int main() {    
    Decode_process_tag processor;
    // teatime, gig-sn01, gig-sn08, monalisa
    string filename = "C:\\Users\\o1r2g\\OneDrive\\Desktop\\cpp_tutorial\\CPPWorkspace\\Refactor\\monalisa.jpg";
    processor.read_data(filename);
    processor.decode();
    
    // system("pause");
    return 0;
}