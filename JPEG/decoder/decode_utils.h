#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <map>
using namespace std;


class Decode_process_tag {
private:
    int max_w = 0;
    int max_h = 0;
    int mcu_y_w = 0;
    int mcu_y_h = 0;
    int mcu_cb_w = 0;
    int mcu_cb_h = 0;
    int mcu_cr_w = 0;
    int mcu_cr_h = 0;
    int block_w = 0;
    int block_h = 0;
    vector <string> rawhex {};
    vector <int> * row_prosessor;
    vector <int> * col_prosessor;    
    vector <int> * tmp_prosessor;     
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
    bool is_sofo_finish = false;
    int sof0_data_p {0};
    int sof0_h {0};
    int sof0_w {0};
    int sof0_c {0};
    vector <int> sof0_c_sampling_h {};  // 2, 1, 1
    vector <int> sof0_c_sampling_v {};  // 2, 1, 1
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
    vector <vector <double>> * img_y;
    vector <vector <double>> * img_cb;
    vector <vector <double>> * img_cr;
    string binary_data {};
    vector <int> blocks_val {};
    vector <double> idct_val {};
    vector <double> rgb_val {};
    // EOI
    bool eoi {false};
public:
    // Attributes
    void decode();
    
    void soi(string & hex_tmp);
    
    void app0(string & hex_tmp, int & read_in_tmp, int & read_in_total);
    
    void com(string & hex_tmp, int & read_in_tmp, int & read_in_total);
    
    void dqt(string & hex_tmp, int & read_in_tmp, int & read_in_total);
    
    void sof0(string & hex_tmp, int & read_in_tmp, int & read_in_total);

    void dht(string & hex_tmp, int & read_in_tmp, int & read_in_total);
    
    void sos(string & hex_tmp, int & read_in_tmp, int & read_in_total, bool & scan_flag);

    void scan(string & hex_tmp, int & read_in_tmp, int & read_in_total, string & bin_tmp);

    
    // Utils
    void read_data(string filename);
    void de_huffman(int & read_in_tmp, int & read_in_total);
    void dpcm(int cur_c, int block_count, int val, bool & dc_y_init, bool & dc_cb_init, bool & dc_cr_init, 
                         int & dc_y, int & dc_cb, int & dc_cr);
    void dqt_mul(const int & q_id);
    void idct(int cur_c, int acc_val, int row_y, int col_y, int row_c, int col_c);
    void ycc2rgb();

    void rc_pos(int cur_c, int block_count, int acc_val, int old_acc_val, int & row_y, int & col_y, int & row_c, int & col_c);  
    void get_dqt_info();
};

