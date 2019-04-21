#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <bitset>

#include "decode_utils.h"
using namespace std;


int main() {
    ifstream f("C:\\Users\\o1r2g\\OneDrive\\Desktop\\cpp_tutorial\\CPPWorkspace\\JPEG\\monalisa.jpg", ios::binary);
    vector <uint8_t> v {istreambuf_iterator<char>{f}, {}};
//    cout << "Read complete, got " << v.size() << " bytes\n";
    
    vector<string> v_hex {};
    for (int i : v) {
        stringstream stream;
        stream << hex << (int)i;
        v_hex.push_back(stream.str());
    }
    Decode_process_tag tag_processor {v_hex, false, false, false};

    tag_processor.process_all_markers();
    
//    tag_processor.get_soi_info();
    tag_processor.get_dqt_info();
//    tag_processor.get_sof0_info();
//    tag_processor.get_dht_info();
//    tag_processor.get_sos_info();
//    tag_processor.get_eoi_info();
//    tag_processor.get_scan_info();
    tag_processor.get_idct_info();
    
    return 0;
}
