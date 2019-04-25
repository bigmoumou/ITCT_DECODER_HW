#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include "decode_utils.h"
#include <map>
#include <bitset>
#include <math.h>
#include <algorithm>
#define PI 3.14159265

# include "easyBMP/EasyBMP.h"
using namespace std;


void Decode_process_tag::read_data(string filename) {
    ifstream f (filename, ios::binary);
    if (!f) { 
        cout << "File not found !" << endl;
        exit(1);
    }
    vector <uint8_t> v {istreambuf_iterator<char>{f}, {}};
    for (int i=0; i < v.size(); i++) {
        stringstream ss;
        ss << hex << (int)v.at(i);
        rawhex.push_back(ss.str());
    }
}

void Decode_process_tag::decode() {
    string hex_tmp {};
    string bin_tmp {};
    bool scan_flag {false};
    int read_in_tmp {0};
    int read_in_total {0};

    for (int i {0}; i < rawhex.size(); i++) {

        if (rawhex.at(i).length() == 1) {
            hex_tmp += ("0" + rawhex.at(i));
        } else {
            hex_tmp += rawhex.at(i);
        }
        
        if (scan_flag) {
            scan(hex_tmp, read_in_tmp, read_in_total, bin_tmp);
        } else {
            if (hex_tmp.length() >= 4) {
                if (read_in_tmp != 0) {
                    read_in_tmp -= 1;
                    continue;
                }
                if (hex_tmp.substr(0, 4) == "ffd8") {
                    // Process SOI marker
                    soi(hex_tmp);         
                } else if (hex_tmp.substr(0, 4) == "ffe0") {
                    // Process APP0 marker
                    app0(hex_tmp, read_in_tmp, read_in_total);
                } else if (hex_tmp.substr(0, 4) == "fffe") {
                    // Process COM marker
                    com(hex_tmp, read_in_tmp, read_in_total);
                } else if (hex_tmp.substr(0, 4) == "ffdb") {
                    // Process DQT marker
                    dqt(hex_tmp, read_in_tmp, read_in_total);
                } else if (hex_tmp.substr(0, 4) == "ffc0") {
                    // Process SOF0 marker
                    sof0(hex_tmp, read_in_tmp, read_in_total);
                    if (is_sofo_finish) {
                        max_w = *max_element(begin(sof0_c_sampling_h), end(sof0_c_sampling_h));
                        max_h = *max_element(begin(sof0_c_sampling_v), end(sof0_c_sampling_v));
                        
                        mcu_y_w = 8 * sof0_c_sampling_h.at(0);
                        mcu_y_h = 8 * sof0_c_sampling_v.at(0);
                        mcu_cb_w = 8 * sof0_c_sampling_h.at(1);
                        mcu_cb_h = 8 * sof0_c_sampling_v.at(1);
                        mcu_cr_w = 8 * sof0_c_sampling_h.at(2);
                        mcu_cr_h = 8 * sof0_c_sampling_v.at(2);
                        
                        block_w = sof0_w / mcu_y_w + (((sof0_w % mcu_y_w) != 0 ) ? 1 : 0);
                        block_h = sof0_h / mcu_y_h + (((sof0_h % mcu_y_h) != 0 ) ? 1 : 0);
                        img_y = new vector <vector <double>> (block_h * mcu_y_h, vector<double>(block_w * mcu_y_w));
                        img_cb = new vector <vector <double>> (block_h * mcu_cb_h, 
                                                                                                        vector<double>(block_w * mcu_cb_w));
                        img_cr = new vector <vector <double>> (block_h * mcu_cr_h, 
                                                                                                        vector<double>(block_w * mcu_cr_w));
                    }
                } else if (hex_tmp.substr(0, 4) == "ffc4") {
                    dht(hex_tmp, read_in_tmp, read_in_total);
                } else if (hex_tmp.substr(0, 4) == "ffda") {
                    sos(hex_tmp, read_in_tmp, read_in_total, scan_flag);
                }
            }
        }
    }
    
    de_huffman(read_in_tmp, read_in_total);
}


void Decode_process_tag::soi(string & hex_tmp) {
    if (hex_tmp == "ffd8")
        is_soi = true;
        hex_tmp = {};
}

void Decode_process_tag::app0(string & hex_tmp, int & read_in_tmp, int & read_in_total) {
    
    if (hex_tmp.length() == 4) {
        if (hex_tmp == "ffe0") {
            read_in_tmp = 1;  // 2 - 1 = 1
        }
    } else if (hex_tmp.length() == 8) {
        // cout << "    [ Debug ] substr is : " << hex_tmp.substr(4, 4) << endl;
        int total_len {0};
        stringstream stream(hex_tmp.substr(4, 4).c_str());
        stream >> hex >> total_len;
        total_len -= 2;
        read_in_tmp = total_len - 1;
        read_in_total = 8 + total_len * 2;
        // cout << "    [ Debug ] total_len is : " << total_len << endl;
    } else if (hex_tmp.length() == read_in_total) {
        // cout << "    [ Debug ] total_len after is : " << hex_tmp.length() << endl;
        // cout << "    [ Debug ] hex_tmp is : " << hex_tmp << endl;
        read_in_total = 0;
        if (hex_tmp.substr(8, 10) == "4a46494600")
            is_app0_tag = true;
            
        int ver {0};
        stringstream maj_ver_stream(hex_tmp.substr(18, 2).c_str());
        maj_ver_stream >> hex >> ver;
        app0_maj_ver = ver;
        // cout << "    [ Debug ] is_app0_tag is : " << is_app0_tag << endl;
        // cout << "    [ Debug ] app0_maj_ver is : " << app0_maj_ver << endl;

        ver = 0;
        stringstream min_ver_stream(hex_tmp.substr(20, 2).c_str());
        min_ver_stream >> hex >> ver;
        app0_min_ver = ver;
        // cout << "    [ Debug ] app0_min_ver is : " << app0_min_ver << endl;
        
        int units {0};
        stringstream units_stream(hex_tmp.substr(22, 2).c_str());
        units_stream >> hex >> units;
        app0_units = units;
        // cout << "    [ Debug ] app0_units is : " << app0_units << endl;
        
        int x_density {0};
        stringstream x_d_stream(hex_tmp.substr(24, 4).c_str());
        x_d_stream >> hex >> x_density;
        app0_x_density = x_density;
        // cout << "    [ Debug ] x_density is : " << x_density << endl;

        int y_density {0};
        stringstream y_d_stream(hex_tmp.substr(28, 4).c_str());
        y_d_stream >> hex >> y_density;
        app0_y_density = y_density;
        // cout << "    [ Debug ] y_density is : " << y_density << endl;
        
        int thumb_w {0};
        stringstream thumb_w_stream(hex_tmp.substr(32, 2).c_str());
        thumb_w_stream >> hex >> thumb_w;
        app0_thumb_w = thumb_w;
        // cout << "    [ Debug ] app0_thumb_w is : " << app0_thumb_w << endl;       
        
         int thumb_h {0};
        stringstream thumb_h_stream(hex_tmp.substr(34, 2).c_str());
        thumb_h_stream >> hex >> thumb_h;
        app0_thumb_h = thumb_h;
        // cout << "    [ Debug ] app0_thumb_h is : " << app0_thumb_h << endl;

        is_app0 = true;
        hex_tmp = {};
    }
}

void Decode_process_tag::com(string & hex_tmp, int & read_in_tmp, int & read_in_total) {
    // cout << "[ Info ] processing COM markers . . ." << endl;
    // cout << "    [ Debug ] hex_tmp is : " << hex_tmp << endl;
    if (hex_tmp.length() == 4) {
        if (hex_tmp == "fffe") {
            read_in_tmp = 1;  // 2 - 1 = 1
            dqt_num += 1;
        }
    } else if (hex_tmp.length() == 8) {
        // cout << "    [ Debug ] substr is : " << hex_tmp.substr(4, 4) << endl;
        int total_len {0};
        stringstream stream(hex_tmp.substr(4, 4).c_str());
        stream >> hex >> total_len;
        total_len -= 2;
        read_in_tmp = total_len - 1;
        read_in_total = 8 + total_len * 2;
        // cout << "    [ Debug ] total_len is : " << total_len << endl;
    } else if (hex_tmp.length() == read_in_total) {
        // cout << "    [ Debug ] total_len after is : " << hex_tmp.length() << endl;
        // cout << "    [ Debug ] read_in_total hex_tmp is : " << hex_tmp << endl;
        
        read_in_total = 0;
        hex_tmp = {};        
    }
        
}

void Decode_process_tag::dqt(string & hex_tmp, int & read_in_tmp, int & read_in_total) {
    // cout << "[ Info ] processing DQT markers . . ." << endl;
    // cout << "    [ Debug ] hex_tmp is : " << hex_tmp << endl;
    if (hex_tmp.length() == 4) {
        if (hex_tmp == "ffdb") {
            read_in_tmp = 1;  // 2 - 1 = 1
        }
    } else if (hex_tmp.length() == 8) {
        // cout << "    [ Debug ] substr is : " << hex_tmp.substr(4, 4) << endl;
        int total_len {0};
        stringstream stream(hex_tmp.substr(4, 4).c_str());
        stream >> hex >> total_len;
        total_len -= 2;
        read_in_tmp = total_len - 1;
        read_in_total = 8 + total_len * 2;
        // cout << "    [ Debug ] total_len is : " << total_len << endl;
    } else if (hex_tmp.length() == read_in_total) {
        // cout << "    [ Debug ] total_len after is : " << hex_tmp.length() << endl;
        // cout << "    [ Debug ] hex_tmp is : " << hex_tmp << endl;
        read_in_total = 0;

        int dqt_num = ((hex_tmp.length() / 2) - 4) / 65;
        for (int i = 0, cur = 8; i < dqt_num; i++, cur+=130) {
            int p {0};
            stringstream p_stream(hex_tmp.substr(cur, 1).c_str());  // 8
            p_stream >> hex >> p;
            dqt_p.push_back(p);

            int q {0};
            stringstream q_stream(hex_tmp.substr(cur+1, 1).c_str());  // 9
            q_stream >> hex >> q;
            dqt_q.push_back(q);
            
            int tmp_num {0};
            vector <int> dqt_m (64, 0);
            for (int k {cur+2}, j {0}; k < (cur+130); k+=2, j++) {  // 10
                stringstream ss;
                ss << hex << hex_tmp.substr(k, 2);
                ss >> tmp_num;
                dqt_m.at(zigzag_m.at(j)) = tmp_num;
            }
            dqt_tables.push_back(dqt_m);
        }
        hex_tmp = {};
    }
}

void Decode_process_tag::sof0(string & hex_tmp, int & read_in_tmp, int & read_in_total) {
    // cout << "[ Info ] processing SOF0 markers . . ." << endl;
    // cout << "    [ Debug ] hex_tmp is : " << hex_tmp << endl;
    if (hex_tmp.length() == 4) {
        if (hex_tmp == "ffc0") {
            read_in_tmp = 1;  // 2 - 1 = 1
        }
    } else if (hex_tmp.length() == 8) {
        // cout << "    [ Debug ] substr is : " << hex_tmp.substr(4, 4) << endl;
        int total_len {0};
        stringstream stream(hex_tmp.substr(4, 4).c_str());
        stream >> hex >> total_len;
        total_len -= 2;
        read_in_tmp = total_len - 1;
        read_in_total = 8 + total_len * 2;
        // cout << "    [ Debug ] total_len is : " << total_len << endl;
    } else if (hex_tmp.length() == read_in_total) {
        // cout << "    [ Debug ] total_len after is : " << hex_tmp.length() << endl;
        // cout << "    [ Debug ] hex_tmp is : " << hex_tmp << endl;
        read_in_total = 0;

        int p {0};
        stringstream p_stream(hex_tmp.substr(8, 2).c_str());
        p_stream >> hex >> p;
        sof0_data_p = p;

        int h {0};
        stringstream h_stream(hex_tmp.substr(10, 4).c_str());
        h_stream >> hex >> h;
        sof0_h = h;

        int w {0};
        stringstream w_stream(hex_tmp.substr(14, 4).c_str());
        w_stream >> hex >> w;
        sof0_w = w;
 
        int c {0};
        stringstream c_stream(hex_tmp.substr(18, 2).c_str());
        c_stream >> hex >> c;
        sof0_c = c;
   
        int cv1h {0};
        stringstream cv1h_stream(hex_tmp.substr(22, 1).c_str());
        cv1h_stream >> hex >> cv1h;
        sof0_c_sampling_h.push_back(cv1h);
        int cv1v {0};
        stringstream cv1v_stream(hex_tmp.substr(23, 1).c_str());
        cv1v_stream >> hex >> cv1v;
        sof0_c_sampling_v.push_back(cv1v);
        int cv1q {0};
        stringstream cv1q_stream(hex_tmp.substr(24, 2).c_str());
        cv1q_stream >> hex >> cv1q;
        sof0_c_qid.push_back(cv1q);
        
        if (c > 1) {
            int cv2h {0};
            stringstream cv2h_stream(hex_tmp.substr(28, 1).c_str());
            cv2h_stream >> hex >> cv2h;
            sof0_c_sampling_h.push_back(cv2h);
            int cv2v {0};
            stringstream cv2v_stream(hex_tmp.substr(29, 1).c_str());
            cv2v_stream >> hex >> cv2v;
            sof0_c_sampling_v.push_back(cv2v);         
            int cv2q {0};
            stringstream cv2q_stream(hex_tmp.substr(30, 2).c_str());
            cv2q_stream >> hex >> cv2q;
            sof0_c_qid.push_back(cv2q);

            int cv3h {0};
            stringstream cv3h_stream(hex_tmp.substr(34, 1).c_str());
            cv3h_stream >> hex >> cv3h;
            sof0_c_sampling_h.push_back(cv3h);
            int cv3v {0};
            stringstream cv3v_stream(hex_tmp.substr(35, 1).c_str());
            cv3v_stream >> hex >> cv3v;
            sof0_c_sampling_v.push_back(cv3v);
            int cv3q {0};
            stringstream cv3q_stream(hex_tmp.substr(36, 2).c_str());
            cv3q_stream >> hex >> cv3q;
            sof0_c_qid.push_back(cv3q);
        }
        if (c > 3) {
            int cv4h {0};
            stringstream cv4h_stream(hex_tmp.substr(40, 1).c_str());
            cv4h_stream >> hex >> cv4h;
            sof0_c_sampling_h.push_back(cv4h);
            int cv4v {0};
            stringstream cv4v_stream(hex_tmp.substr(41, 1).c_str());
            cv4v_stream >> hex >> cv4v;
            sof0_c_sampling_v.push_back(cv4v);
            int cv4q {0};
            stringstream cv4q_stream(hex_tmp.substr(42, 2).c_str());
            cv4q_stream >> hex >> cv4q;
            sof0_c_qid.push_back(cv4q);
        }
        
        is_sofo_finish = true;
        hex_tmp = {};
    }
}

void Decode_process_tag::dht(string & hex_tmp, int & read_in_tmp, int & read_in_total) {
    // cout << "[ Info ] processing DHT markers . . ." << endl;
    // cout << "    [ Debug ] hex_tmp is : " << hex_tmp << endl;
    if (hex_tmp.length() == 4) {
        if (hex_tmp == "ffc4") {
            read_in_tmp = 1;  // 2 - 1 = 1
            dht_num += 1;
        }
    } else if (hex_tmp.length() == 8) {
        // cout << "    [ Debug ] substr is : " << hex_tmp.substr(4, 4) << endl;
        int total_len = 0;
        stringstream stream(hex_tmp.substr(4, 4).c_str());
        stream >> hex >> total_len;
        total_len -= 2;
        read_in_tmp = total_len - 1;
        read_in_total = 8 + total_len * 2;
        // cout << "    [ Debug ] total_len is : " << total_len << endl;
    } else if (hex_tmp.length() == read_in_total) {
        // cout << "    [ Debug ] hex_tmp.length() is : " << hex_tmp.length() << endl;
        // cout << "    [ Debug ] hex_tmp is : " << hex_tmp << endl;
        read_in_total = 0;
        int cur = 8;

        for (int dht_cur = 8; dht_cur < hex_tmp.length(); dht_cur += (cur - 8)) {
            int htc = 0;
            string htc_s {};
            stringstream htc_stream(hex_tmp.substr(cur, 1).c_str());  // 8
            htc_stream >> hex >> htc;
            if (htc == 0) {
                htc_s = "DC_";
            } else if (htc == 1) {
                htc_s = "AC_";
            }
            dht_ht_c.push_back(htc);
            cur += 1;

            int htid = 0;
            stringstream htid_stream(hex_tmp.substr(cur, 1).c_str());  // 9
            htid_stream >> hex >> htid;
            dht_ht_id.push_back(htid);
            cur += 1;
            
            vector <int> table_count (16, 0); 
            // 10
            for (int i = cur, j = 0; i < (cur + 32); i+=2, j++) {
                int tmp_c = 0;
                stringstream c_stream(hex_tmp.substr(i, 2).c_str());
                c_stream >> hex >> tmp_c;
                table_count.at(j) = tmp_c;
            }
            cur += 32;

            // 42
            vector <string> table_value {};
            for (int i {0}; i < table_count.size(); i++) {
                if (table_count.at(i) != 0) {
                    if (htc_s == "DC_") {
                        for (int j = cur; j < cur + table_count.at(i) * 2 ; j+=2) {
                            string tmp_v = "";
                            stringstream v_stream(hex_tmp.substr(j, 2).c_str());
                            v_stream >> hex >> tmp_v;
                            table_value.push_back(tmp_v);
                        }
                    } else if (htc_s == "AC_") {
                        for (int j = cur; j < cur + table_count.at(i) * 2 ; j+=2) {
                            string tmp_v = "";
                            string first_v = "";
                            string second_v = "";
                            stringstream f_stream(hex_tmp.substr(j, 1).c_str());
                            f_stream >> hex >> first_v;
                            tmp_v += first_v;
                            stringstream s_stream(hex_tmp.substr(j+1, 1).c_str());
                            s_stream >> hex >> second_v;
                            tmp_v += second_v;
                            
                            table_value.push_back(tmp_v);
                        }
                    }
                    cur += table_count.at(i) * 2;
                }
            }

            int tmp_int = 0;
            int tmp_num = 0;
            bool init_flag = true;
            map <string, string> tmp_map;

            int lsb_mask = 1;
            for (int i = 0; i < table_count.size(); i++) {
                for (int j = 0; j < table_count.at(i); j++) {
                    int tmp_key {tmp_int};
                    string str_key = "";
                    for (int k = 0; k < i + 1; k += 1) {
                        str_key = to_string((tmp_key & lsb_mask)) + str_key;
                        tmp_key >>= 1;
                    }
                    tmp_map[str_key] = table_value.at(tmp_num);

                    tmp_num += 1;
                    tmp_int += 1;
                }
                tmp_int <<= 1;
            }
            
            // "DC_0" "DC_1" "AC_0" "AC_1"
            dht_map[htc_s + to_string(htid)] = tmp_map;
        }
        hex_tmp = {};
    }
}

void Decode_process_tag::sos(string & hex_tmp, int & read_in_tmp, int & read_in_total, bool & scan_flag) {
    // cout << "[ Info ] processing SOS markers . . ." << endl;
    // cout << "    [ Debug ] hex_tmp is : " << hex_tmp << endl;
    if (hex_tmp.length() == 4) {
        if (hex_tmp == "ffda") {
            read_in_tmp = 1;  // 2 - 1 = 1
            dqt_num += 1;
        }
    } else if (hex_tmp.length() == 8) {
        // cout << "    [ Debug ] substr is : " << hex_tmp.substr(4, 4) << endl;
        int total_len {0};
        stringstream stream(hex_tmp.substr(4, 4).c_str());
        stream >> hex >> total_len;
        total_len -= 2;
        read_in_tmp = total_len - 1;
        read_in_total = 8 + total_len * 2;
        // cout << "    [ Debug ] total_len is : " << total_len << endl;
    } else if (hex_tmp.length() == read_in_total) {
        // cout << "    [ Debug ] total_len after is : " << hex_tmp.length() << endl;
        // cout << "    [ Debug ] hex_tmp is : " << hex_tmp << endl;
        read_in_total = 0;

        int c {0};
        stringstream c_stream(hex_tmp.substr(8, 2).c_str());
        c_stream >> hex >> c;
        sos_c = c;
        
        int c1 {0};
        map <string, int> c1_table_map;
        stringstream c1dc_stream(hex_tmp.substr(12, 1).c_str());
        c1dc_stream >> hex >> c1;
        c1_table_map["DC_"] = c1;
        c1 = 0;
        stringstream c1ac_stream(hex_tmp.substr(13, 1).c_str());
        c1ac_stream >> hex >> c1;
        c1_table_map["AC_"] = c1;        
        sos_c_table_use.push_back(c1_table_map);

        int cursor {14};

        if (sos_c > 1) {
            int c2 {0};
            map <string, int> c2_table_map;
            stringstream c2dc_stream(hex_tmp.substr(16, 1).c_str());
            c2dc_stream >> hex >> c2;
            c2_table_map["DC_"] = c2;
            c2 = 0;
            stringstream c2ac_stream(hex_tmp.substr(17, 1).c_str());
            c2ac_stream >> hex >> c2;
            c2_table_map["AC_"] = c2;        
            sos_c_table_use.push_back(c2_table_map);

            int c3 {0};
            map <string, int> c3_table_map;
            stringstream c3dc_stream(hex_tmp.substr(20, 1).c_str());
            c3dc_stream >> hex >> c3;
            c3_table_map["DC_"] = c3;
            c3 = 0;
            stringstream c3ac_stream(hex_tmp.substr(21, 1).c_str());
            c3ac_stream >> hex >> c3;
            c3_table_map["AC_"] = c3;        
            sos_c_table_use.push_back(c3_table_map);
            
            cursor = 22;
        }
        if (c > 3) {
            int c4 {0};
            map <string, int> c4_table_map;
            stringstream c4dc_stream(hex_tmp.substr(24, 1).c_str());
            c4dc_stream >> hex >> c4;
            c4_table_map["DC_"] = c4;
            c4 = 0;
            stringstream c4ac_stream(hex_tmp.substr(25, 1).c_str());
            c4ac_stream >> hex >> c4;
            c4_table_map["AC_"] = c4;        
            sos_c_table_use.push_back(c4_table_map);
            
            cursor = 26;
        }
        // Ignorable 3 bytes (JPEG baseline)
        ignorable_bytes = hex_tmp.substr(cursor, 6);

        hex_tmp = {};
        scan_flag = true;
    }
}

void Decode_process_tag::scan(string & hex_tmp, int & read_in_tmp, int & read_in_total, string & bin_tmp) {
    if (!bin_tmp.empty()) {
        if (hex_tmp == "00") {
            binary_data += bin_tmp;
            bin_tmp = "";
            hex_tmp = "";
        } else if (hex_tmp == "d9") {
            eoi = true;
            bin_tmp = "";
            hex_tmp = "";
        }
    } else if (hex_tmp == "ff") {
        string ff_str {"0xff"};
        stringstream ss;
        ss << hex << ff_str;
        unsigned n;
        ss >> n;
        bitset<8> b(n);
        bin_tmp += b.to_string();
        hex_tmp = "";
    } else {
        string ff_str {"0x" + hex_tmp};
        stringstream ss;
        ss << hex << ff_str;
        unsigned n;
        ss >> n;
        bitset<8> b(n);
        binary_data += b.to_string();
        hex_tmp = "";
    }
}

void Decode_process_tag::de_huffman(int & read_in_tmp, int & read_in_total) {
    bool dc_y_init = false;
    bool dc_cb_init = false;
    bool dc_cr_init = false;
    int dc_y = 0;
    int dc_cb = 0;
    int dc_cr = 0;
   
    int y_num = sof0_c_sampling_h.at(0) * sof0_c_sampling_v.at(0);
    int cb_num = sof0_c_sampling_h.at(1) * sof0_c_sampling_v.at(1);
    int cr_num = sof0_c_sampling_h.at(2) * sof0_c_sampling_v.at(2);
   
    string str_tmp = "";
    int cur_c = 0;                 // YCC : 0, 1, 2
    string dcac = "DC_";   // "DC", "AC"
    int h_id = 0;
    int block_count = 0; // 0-63
    int mcu_count = 0;   // 0-255
    bool is_neg = false;
    bool keep_search = true;
    
    int row_y = 0;
    int col_y = 0;
    int row_c = 0;
    int col_c = 0;
    int acc_val = 0;
    int old_acc_val = 0;
    int _c = 0;

    row_prosessor = new vector <int> (64, 0);     
    col_prosessor = new vector <int> (64, 0);        
    tmp_prosessor = new vector <int> (64, 0);     
    for (char s: binary_data) {
        old_acc_val = acc_val;
        
        str_tmp += s;
        if (read_in_tmp > 0) {
            read_in_tmp -= 1;
            continue;
        }
        h_id = sos_c_table_use.at(cur_c)[dcac];
        // cout << endl << "----- using table : " << dcac + to_string(h_id) << " -----" << endl;
        // cout << "Run : " << _c + 1 << " >> key : " << str_tmp << endl;

        // read-in condition satisfied
        if (str_tmp.length() == read_in_total) {
            // cout << "     -> ** read in str_tmp is : " << str_tmp << endl;
            // 1's complement
            if (str_tmp.substr(0, 1) == "0") {
                is_neg = true;
                for (int i = 0; i < str_tmp.length(); i++) {
                    if (str_tmp.substr(i, 1) == "0") {
                        str_tmp.replace(i, 1, "1");
                    } else if (str_tmp.substr(i, 1) == "1") {
                        str_tmp.replace(i, 1, "0");
                    }
                }
            }       
            // negative after 1's complement
            int val =  stoi(str_tmp, nullptr, 2);
            if (is_neg) {
                val = - val;
            }
            
            if (dcac == "DC_") {
                dpcm(cur_c, block_count, val, dc_y_init, dc_cb_init, dc_cr_init, dc_y, dc_cb, dc_cr);
                acc_val += 1;
                rc_pos(cur_c, block_count, acc_val, old_acc_val, row_y, col_y, row_c, col_c);
            } else {
                switch (cur_c) {
                    case 0:
                        tmp_prosessor->at(block_count) = val;
                        acc_val += 1;
                        rc_pos(cur_c, block_count, acc_val, old_acc_val, row_y, col_y, row_c, col_c);
                        break;
                    case 1:
                        tmp_prosessor->at(block_count) = val;
                        acc_val += 1;
                        rc_pos(cur_c, block_count, acc_val, old_acc_val, row_y, col_y, row_c, col_c);
                        break;
                    case 2:
                        tmp_prosessor->at(block_count) = val;
                        acc_val += 1;
                        rc_pos(cur_c, block_count, acc_val, old_acc_val, row_y, col_y, row_c, col_c);
                        break;
                }
            }
            // cout << "     -> " << str_tmp << " >>> " << val << endl;            
            read_in_total = 0;
            str_tmp = "";
            is_neg = false;
            block_count += 1;
            mcu_count += 1;
            // cout << "     -> block_count : " << block_count << " -> mcu_count : " << mcu_count << endl;
        }

        if (keep_search) {
            map<string, string>::iterator iter;
            iter = dht_map[dcac + to_string(h_id)].find(str_tmp);
            if (iter !=dht_map[dcac + to_string(h_id)].end()) {
                if (dcac == "DC_") {
                    int dc_val {0};
                    stringstream ss(iter->second.c_str());
                    ss >> hex >> dc_val;
                    if (dc_val == 0) {
                        read_in_tmp = 0;
                        read_in_total = 0;
                        dpcm(cur_c, block_count, dc_val, dc_y_init, dc_cb_init, dc_cr_init, dc_y, dc_cb, dc_cr);
                        acc_val += 1;
                        rc_pos(cur_c, block_count, acc_val, old_acc_val, row_y, col_y, row_c, col_c);
                        block_count += 1;
                        mcu_count += 1;
                        // cout << "     -> block_count : " << block_count << " -> mcu_count : " << mcu_count << endl;
                        keep_search = true;
                    } else {
                        read_in_tmp = dc_val - 1;
                        read_in_total = dc_val;
                        keep_search = false;                        
                    }
                    // cout << "     -> " << iter->second << endl;
                    // cout << "     -> " << dc_val << endl;
                } else if (dcac == "AC_") {
                    int ac_val_f {0};
                    int ac_val_s {0};
                    stringstream fs(iter->second.substr(0, 1).c_str());
                    fs >> hex >> ac_val_f;
                    // add 0 with 1st value
                    if (ac_val_f != 0) {
                        for (int i = 0; i < ac_val_f; i++) {
                            tmp_prosessor->at(block_count) = 0;
                            acc_val += 1;
                            rc_pos(cur_c, block_count, acc_val, old_acc_val, row_y, col_y, row_c, col_c);
                            block_count += 1;
                            mcu_count += 1;
                            // cout << "     -> block_count : " << block_count << " -> mcu_count : " << mcu_count << endl;
                        }
                    }
                    
                    stringstream ss(iter->second.substr(1, 1).c_str());
                    ss >> hex >> ac_val_s;
                    
                    // Process "00" for AC
                    if ((ac_val_f == 0) && (ac_val_s == 0)) {
                        int add_num = (63 - block_count + 1);
                        for (int i = 0; i < add_num; i++) {
                            tmp_prosessor->at(block_count) = 0;
                            acc_val += 1;
                            rc_pos(cur_c, block_count, acc_val, old_acc_val, row_y, col_y, row_c, col_c);
                            block_count += 1;
                            mcu_count += 1;
                        }
                        read_in_total = 0;
                        str_tmp = "";
                        is_neg = false;
                        // block_count += add_num;
                        // mcu_count += add_num;
                        keep_search = true;        
                        // cout << "     -> block_count : " << block_count << " -> mcu_count : " << mcu_count << endl;
                    } else if ((ac_val_f == 15) && (ac_val_s == 0)) {
                            tmp_prosessor->at(block_count) = 0;
                            acc_val += 1;
                            rc_pos(cur_c, block_count, acc_val, old_acc_val, row_y, col_y, row_c, col_c);
                            block_count += 1;
                            mcu_count += 1;
                            // cout << "     -> process for F0, add one more 0 element";
                            // cout << "     -> block_count : " << block_count << " -> mcu_count : " << mcu_count << endl;
                    } else if (ac_val_s == 0) {
                        keep_search = true;
                    } else {
                        read_in_tmp = ac_val_s - 1;
                        read_in_total = ac_val_s;
                        keep_search = false;
                    }
                    // cout << "     -> " << iter->second << endl;
                    // cout << "     -> first : " << ac_val_f << endl;
                    // cout << "     -> second : " << ac_val_s << endl;
                }
                str_tmp = "";
            }            
        }
        
        _c += 1;

        // re-open search
        if ((!keep_search) && (read_in_total == 0)) {
            keep_search = true;
        }

        if ((1 <= block_count) && (block_count <= 63)) {
            dcac = "AC_";
        } else if (block_count >= 64) {
            // DQT
            switch (cur_c) {
                // Y
                case 0:
                    dqt_mul(sof0_c_qid.at(0));
                    break;
                // Cb
                case 1:
                    dqt_mul(sof0_c_qid.at(1));
                    break;
                // Cr
                case 2:
                    dqt_mul(sof0_c_qid.at(2));
                    break;
            }            
            // IDCT
            idct(cur_c, acc_val, row_y, col_y, row_c, col_c);
             // re - init
            block_count = 0;
            dcac = "DC_"; 
        }
       
       // 0 - 64 * 6
        if ((0 <= mcu_count) && (mcu_count <= (64 * y_num - 1))) {
            cur_c = 0;
        } else if (((64 * y_num) <= mcu_count) && (mcu_count <= (64 * (y_num + cb_num) - 1))) {
            cur_c = 1;
        } else if (((64 * (y_num + cb_num)) <= mcu_count) && (mcu_count <= (64 * (y_num + cb_num + cr_num) - 1))) {
            cur_c = 2;
        } else if (mcu_count >= (64 * (y_num + cb_num + cr_num))) {
            // re - init
            mcu_count = 0;
            cur_c = 0;
        }
    }
    ycc2rgb();
}


// Utils
void Decode_process_tag::dpcm(int cur_c, int block_count, int val, bool & dc_y_init, bool & dc_cb_init, bool & dc_cr_init,
                                                              int & dc_y, int & dc_cb, int & dc_cr) {
    if ((cur_c == 0) && !dc_y_init) {
        dc_y = val;
        dc_y_init = true;
        tmp_prosessor->at(block_count) = dc_y;
    } else if ((cur_c == 0) && dc_y_init) {
        dc_y += val;
        tmp_prosessor->at(block_count) = dc_y;
    } else if ((cur_c == 1) && !dc_cb_init) {
        dc_cb = val;
        dc_cb_init = true;
        tmp_prosessor->at(block_count) = dc_cb;
    } else if ((cur_c == 1) && dc_cb_init) {
        dc_cb += val;
        tmp_prosessor->at(block_count) = dc_cb;
    } else if ((cur_c == 2) && !dc_cr_init) {
        dc_cr = val;
        dc_cr_init = true;
        tmp_prosessor->at(block_count) = dc_cr;
    } else if ((cur_c == 2) && dc_cr_init) {
        dc_cr += val;
        tmp_prosessor->at(block_count) = dc_cr;
    }     
}

void Decode_process_tag::dqt_mul(const int & q_id) {
    vector <int> q_t = dqt_tables.at(q_id);
    vector <int> zigzag_b (64, 0);
    for (int i = 0; i < 64; i++) {
        zigzag_b.at(zigzag_m.at(i)) = tmp_prosessor->at(i);
    }

    for (int i = 0; i < 64; i++) {
        tmp_prosessor->at(i) = q_t.at(i) * zigzag_b.at(i);
        // tmp_prosessor->at(i) = q_t.at(i) * tmp_prosessor->at(i);
    }
}

void Decode_process_tag::idct(int cur_c, int acc_val, int row_y, int col_y, int row_c, int col_c) {
    vector <double> idct_prosessor (64, 0);
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            double idct_val = 0;
             for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    double tmp = tmp_prosessor->at((i * 8) + j) * cos(((2 * x + 1) * i * PI) / (16)) * cos(((2 * y + 1) * j * PI) / (16));
                    if (i == 0) {
                        tmp /= sqrt(2);
                    }
                    if (j == 0) {
                        tmp /= sqrt(2);
                    }
                    idct_val += tmp;
                }
            }
            idct_val /= 4;
            idct_prosessor.at(x * 8 + y) = idct_val;
        }
    }

    switch (cur_c) {
        case 0:
            for (int i = 0; i < 64; i++) {
                img_y->at(row_prosessor->at(i)).at(col_prosessor->at(i)) = idct_prosessor.at(i);
            }
        break;
        case 1:
            for (int i = 0; i < 64; i++) {
                img_cb->at(row_prosessor->at(i)).at(col_prosessor->at(i)) = idct_prosessor.at(i);
            }
        break;
        case 2:
            for (int i = 0; i < 64; i++) {
                img_cr->at(row_prosessor->at(i)).at(col_prosessor->at(i)) = idct_prosessor.at(i);
            }
        break;
    }
}

void Decode_process_tag::rc_pos(int cur_c, int block_count, int acc_val, int old_acc_val, int & row_y, int & col_y,
                                                               int & row_c, int & col_c) {
    // cout << "old / acc_val : " << old_acc_val << "  <  " << acc_val << endl;
    int y_num = sof0_c_sampling_h.at(0) * sof0_c_sampling_v.at(0);
    int cb_num = sof0_c_sampling_h.at(1) * sof0_c_sampling_v.at(1);
    int cr_num = sof0_c_sampling_h.at(2) * sof0_c_sampling_v.at(2);    
    
    switch(cur_c){
        case 0:
        if (acc_val > old_acc_val) {
        int _j = (acc_val - 1) % (64 * (y_num + cb_num + cr_num));  // 64 * 6
        
        if ((sof0_c_sampling_h.at(0) == 2) && (sof0_c_sampling_v.at(0) == 2)) {
            // 4 : 1 : 1
            if (((_j) % (64 * sof0_c_sampling_h.at(0)) == 0) && (_j != 0)) {    // if (((_j) % 128 == 0) && (_j != 0))
                col_y -= (8 * sof0_c_sampling_h.at(0)) - 1;
                row_y += 1;
            } else if (((_j) % 64 == 0) && (_j != 0)) {
                col_y += 1;
                row_y -= 7;
            } else if ((_j)  % 8 == 0 && (_j != 0)) {
                col_y -= 7;
                row_y += 1;
            } else if (_j != 0) {
                col_y += 1;
            }            
        } else if ((sof0_c_sampling_h.at(0) == 2) && (sof0_c_sampling_v.at(0) == 1)) {
            // 2 : 1 : 1 - h2
            if (((_j) % 64 == 0) && (_j != 0)) {
                col_y += 1;
                row_y -= 7;
            } else if ((_j)  % 8 == 0 && (_j != 0)) {
                col_y -= 7;
                row_y += 1;
            } else if (_j != 0) {
                col_y += 1;
            } 
        } else if ((sof0_c_sampling_h.at(0) == 1) && (sof0_c_sampling_v.at(0) == 2)) {
            // 2 : 1 : 1 - v2
            if (((_j) % 64 == 0) && (_j != 0)) {
                col_y -= 7;
                row_y += 1;
            } else if ((_j)  % 8 == 0 && (_j != 0)) {
                col_y -= 7;
                row_y += 1;
            } else if (_j != 0) {
                col_y += 1;
            }      
        } else if ((sof0_c_sampling_h.at(0) == 1) && (sof0_c_sampling_v.at(0) == 1)) {
            // 1 : 1 : 1
            if ((_j)  % 8 == 0 && (_j != 0)) {
                col_y -= 7;
                row_y += 1;
            } else if (_j != 0) {
                col_y += 1;
            }      
        }
        
     // cout << "[ Y ] (r, c) : " << row_y << ", " << col_y << "     acc_val : " << acc_val << endl;   
     row_prosessor->at(block_count) = row_y;
     col_prosessor->at(block_count) = col_y;
    
        if ((sof0_c_sampling_h.at(0) == 2) && (sof0_c_sampling_v.at(0) == 2)) {
            // 4 : 1 : 1
            if ((_j == 255) && (col_y == (block_w * mcu_y_w - 1))) {   // 18 * 16 -1
                col_y = 0;
                row_y += 1;
            } else if (_j == 255) {
                col_y += 1;
                row_y -= 15;
            }
        } else if ((sof0_c_sampling_h.at(0) == 2) && (sof0_c_sampling_v.at(0) == 1)) {
            // 2 : 1 : 1 - h2
            if ((_j == 127) && (col_y == (block_w * mcu_y_w - 1))) {   // 18 * 16 -1
                col_y = 0;
                row_y += 1;
            } else if (_j == 127) {
                col_y += 1;
                row_y -= 7;
            }
        } else if ((sof0_c_sampling_h.at(0) == 1) && (sof0_c_sampling_v.at(0) == 2)) {
            // 2 : 1 : 1 - v2
            if ((_j == 127) && (col_y == (block_w * mcu_y_w - 1))) {   // 18 * 16 -1
                col_y = 0;
                row_y += 1;
            } else if (_j == 127) {
                col_y += 1;
                row_y -= 15;
            }
        } else if ((sof0_c_sampling_h.at(0) == 1) && (sof0_c_sampling_v.at(0) == 1)) {
            // 1 : 1 : 1
            if ((_j == 63) && (col_y == (block_w * mcu_y_w - 1))) {   // 18 * 16 -1
                col_y = 0;
                row_y += 1;
            } else if (_j == 63) {
                col_y += 1;
                row_y -= 7;
            }
        }  
    }
    break;
    case 1:
        if (acc_val > old_acc_val) {
        int _j = (acc_val - 1) % (64 * ((y_num + cb_num + cr_num))) - (64 * y_num);    // int _j = (acc_val - 1) % (64 * 6) - (64 * 4);  // 0 - 63
        if (((_j) % 64 == 0) && (_j != 0)) {
            col_c += 1;
            row_c -= 7;
        } else if ((_j)  % 8 == 0 && (_j != 0)) {
            col_c -= 7;
            row_c += 1;
        } else if (_j != 0) {
            col_c += 1;
        }
        // cout << "[ C ] (r, c) : " << row_c << ", " << col_c << "     acc_val : " << acc_val << "    j : " << _j << "    ";

        row_prosessor->at(block_count) = row_c;
        col_prosessor->at(block_count) = col_c;
        
        if ((_j == 63) && (col_c == (block_w * mcu_cb_w - 1))) {  // 18 * (16 / 2) - 1
            col_c = 0;
            row_c += 1;
        } else if (_j == 63) {
            col_c += 1;
            row_c -= 7;
        }
    }
    case 2:
    break;
    }
        
}

void Decode_process_tag::ycc2rgb() {
    // Use img_y for indexing
    int img_y_w = img_y->at(0).size();
    int img_y_h = img_y->size();
    int max_hs = *max_element(begin(sof0_c_sampling_h), end(sof0_c_sampling_h));
    int max_vs = *max_element(begin(sof0_c_sampling_v), end(sof0_c_sampling_v));
    BMP Bmp_image;
    Bmp_image.SetSize(img_y_w, img_y_h);
    for (int r = 0; r < img_y_h; r++) {
        for (int c = 0; c < img_y_w; c++) {
            // For Cb
            int cb_r = r * sof0_c_sampling_v.at(1) / max_vs;
            int cb_c = c * sof0_c_sampling_h.at(1) / max_hs;
            // For Cr
            int cr_r = r * sof0_c_sampling_v.at(2) / max_vs;
            int cr_c = c * sof0_c_sampling_h.at(2) / max_hs;

            double Y = img_y->at(r).at(c);
            double Cb = img_cb->at(cb_r).at(cb_c);
            double Cr = img_cr->at(cr_r).at(cr_c);

            double R = Y  + (1.40168806 * Cr) + 128;
            double G = Y - (0.34430587 * Cb) - (0.71368677 * Cr) + 128;
            double B = Y + (1.77530819 * Cb) + 128;

            // Process range to 0 - 255
            if (R < 0) {
                R = 0;
            } else if (R > 255) {
                R = 255;
            }
            if (G < 0) {
                G = 0;
            } else if (G > 255) {
                G = 255;
            }
            if (B < 0) {
                B = 0;
            } else if (B > 255) {
                B = 255;
            }

             Bmp_image (c, r) -> Red = R;
             Bmp_image (c, r) -> Green = G;
             Bmp_image (c, r) -> Blue = B;
             Bmp_image (c, r) -> Alpha = 0;            
        }
    }
    Bmp_image.WriteToFile("JPEG.bmp");
}


// Utils
void Decode_process_tag::get_dqt_info() {
    cout << "    [ Debug ] showing DQT info . . . " << dqt_num << endl;
    cout << "    [ Debug ] total DQT table number is : " << dqt_num << endl;
    for (int e: dqt_p)
        cout <<  "    [ Debug ] each P of DQT is : " << e << endl;
    for (int e: dqt_q)
        cout <<  "    [ Debug ] each id of DQT is : " << e << endl;
    cout << "    [ Debug ] dqt_ms size is : " << dqt_tables.size() << endl;
    for (auto e: dqt_tables) {
        for (int i {0}; i < 64; i++) {
            cout << e.at(i) << "  ";
            if ((i+1) % 8 == 0)
                 cout << endl;
        };
    }; 
};