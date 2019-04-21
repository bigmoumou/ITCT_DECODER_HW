#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <map>
using namespace std;


class Decode_process_tag {
private:
     std::vector<string> rawhexv {};
     // SOI
     bool is_soi {false};
     // APP0
     bool is_app0 {false};
     bool is_app0_tag {false};
     int app0_maj_ver {0};
     int app0_min_ver {0};
     int app0_units {0};
     int app0_x_density {0};
     int app0_y_density {0};
     int app0_thumb_w {0};
     int app0_thumb_h {0};
     // DQT
     int dqt_num {0};
     vector <int> dqt_p {};
     vector <int> dqt_q {}; // 0, 1
     vector <vector <int>> dqt_tables {};
     vector <int> zigzag_m {0, 1, 8, 16, 9, 2, 3, 10,
                                                 17, 24, 32, 25, 18, 11, 4, 5,
                                                 12, 19, 26, 33, 40, 48, 41, 34,
                                                 27, 20, 13, 6, 7, 14, 21, 28,
                                                 35, 42, 49, 56, 57, 50, 43, 36,
                                                 29, 22, 15, 23, 30, 37, 44, 51,
                                                 58, 59, 52, 45, 38, 31, 39, 46,
                                                 53, 60, 61, 54, 47, 55, 62, 63};
    // SOF0
    int sof0_data_p {0};
    int sof0_h {0};
    int sof0_w {0};
    int sof0_c {0};
    vector <int> sof0_c_sampling_h {};
    vector <int> sof0_c_sampling_v {};
    vector <int> sof0_c_qid {};
    // DHT
    int dht_num {0};
    vector <int> dht_ht_c {};
    vector <int> dht_ht_id {};
    map<string, map<string, string>> dht_map {};
    // SOS
    int sos_c {0};
    vector <map<string, int>> sos_c_table_use {};
    string ignorable_bytes {};
    // SCAN
    string mcus {};
    vector <int> blocks_val {};
    vector <double> idct_val {};
    vector <double> rgb_val {};
    // EOI
    bool eoi {false};
public:
    // Attributes
    bool process_all_markers();
    
    void soi(string & hex_tmp);
    void get_soi_info();
    
    void app0(string & hex_tmp, int & read_in_tmp, int & read_in_total);
    
    void dqt(string & hex_tmp, int & read_in_tmp, int & read_in_total);
    void get_dqt_info();
    
    void sof0(string & hex_tmp, int & read_in_tmp, int & read_in_total);
    void get_sof0_info();

    void dht(string & hex_tmp, int & read_in_tmp, int & read_in_total);
    void get_dht_info();
    
    void sos(string & hex_tmp, int & read_in_tmp, int & read_in_total, bool & scan_flag);
    void get_sos_info();
    
    void scan(string & hex_tmp, int & read_in_tmp, int & read_in_total, string & bin_tmp);
    void get_eoi_info();
    void get_scan_info();
    void get_idct_info();
    
    // Utils
    void dpcm(int cur_c, int val, bool & dc_y_init, bool & dc_cb_init, bool & dc_cr_init, 
                         int & dc_y, int & dc_cb, int & dc_cr);
    void dqt_mul(const int & q_id);
    void d_zigzag();
    void idct();
    void rebuild_ycc(const int & cur_c);
    void ycc2rgb();
    
    // Overloaded Constructors
    Decode_process_tag();
    Decode_process_tag(std::vector<string> rawhexv_val, bool is_soi_val, bool is_app0_val,
                                            bool is_app0_tag_val);
};

