#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include "decoder_utils.h"
#include <map>
#include <bitset>
#include <deque>
#include <math.h>
#include <algorithm>
#define PI 3.14159265

#include "easyBMP/EasyBMP.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

// Init Decoder
Decoder::Decoder() : dct_zz(64, 0), pattern_code(6, 0), dct_recon(8, vector<int>(8, 0)) {
    // init idct_result
    idct_table.resize(8, vector<double> (8, 0));
    // idct_result;
    for (int i=0; i<8; i++) {
        for (int j=0; j<8; j++) {
            if (j == 0) {
                idct_table.at(i).at(j) = cos(((2 * i + 1) * j * PI) / (16)) / sqrt(2) / 2;
            } else {
                idct_table.at(i).at(j) = cos(((2 * i + 1) * j * PI) / (16)) / 2;
            }
        }
    }
    // init pel pasr RGB
    pel_past_R = vector<vector<int>>(240, vector<int>(320, 0));
    pel_past_G = vector<vector<int>>(240, vector<int>(320, 0));
    pel_past_B = vector<vector<int>>(240, vector<int>(320, 0));
    pel_future_R = vector<vector<int>>(240, vector<int>(320, 0));
    pel_future_G = vector<vector<int>>(240, vector<int>(320, 0));
    pel_future_B = vector<vector<int>>(240, vector<int>(320, 0));
    // init pel RGB
    pel_R = vector<vector<int>>(240, vector<int>(320, 0));
    pel_G = vector<vector<int>>(240, vector<int>(320, 0));
    pel_B = vector<vector<int>>(240, vector<int>(320, 0));
    pel_tmp_R = vector<vector<int>>(240, vector<int>(320, 0));
    pel_tmp_G = vector<vector<int>>(240, vector<int>(320, 0));
    pel_tmp_B = vector<vector<int>>(240, vector<int>(320, 0));
}

// Main functions
void Decoder::read_data(string filename) {
    ifstream f (filename, ios::binary);
    if (!f) {
        cout << "File not found !" << endl;
        exit(1);
    }
    abs_buf = {istreambuf_iterator<char>{f}, {}};
}

void Decoder::video_sequence() {
    next_start_code();
    if (is_next_start_code(0xb3)) {
        do {
            sequence_header();
            do {
            group_of_pictures();
            } while (is_next_start_code(0xb8));
        } while (is_next_start_code(0xb3));  
    }

    bool isend = is_next_start_code(0xb7);
    if (isend) {
        cout << "Sequence End Code" << endl;
    }
}

void Decoder::sequence_header() {
    seq_h_code = read_bits(32);
    h_size = read_bits(12);
    v_size = read_bits(12);
    p_a_r = read_bits(4);
    p_r = read_bits(4);
    b_r = read_bits(18);
    m_b = read_bits(1);
    vbv_buffer_size = read_bits(10);
    constrained_parameter_flag = read_bits(1);
    load_intra_quantizer_matrix = read_bits(1);
    load_non_intra_quantizer_matrix = read_bits(1);   
    mb_width = h_size / 16;
    mb_height = v_size / 16;

    // debug
//    print_hex(seq_h_code);
//    print_hex(h_size);
//    print_hex(v_size);
//    print_hex(p_a_r);
//    print_hex(p_r);
//    print_hex(b_r);
//    print_hex(m_b);
//    print_hex(vbv_buffer_size);
//    print_hex(constrained_parameter_flag);
//    print_hex(load_intra_quantizer_matrix);
//    print_hex(load_non_intra_quantizer_matrix);
    
    if ( load_intra_quantizer_matrix ) {
        load_intra_quant();
    }
    if ( load_non_intra_quantizer_matrix ) {
        load_non_intra_quant();
    }
    next_start_code();
}

void Decoder::group_of_pictures() {
    // pic_num = 0;
    gop_start_code = read_bits(32);
    time_code = read_bits(25);
    closed_gop = read_bits(1);
    broken_link = read_bits(1);
    
    // debug
//    print_hex(gop_start_code);
//    print_hex(time_code);
//    print_hex(closed_gop);
//    print_hex(broken_link);
    
    next_start_code();
    
    if ((zero_byte_flag == true) && (zero_byte == 0)) {
        // Picture start code '00000100'
        buf = 0;
        buf_cursor = 0;
        zero_byte_flag = false;
        do {
            picture();
        } while (is_next_start_code(0));
    }
}

void Decoder::picture() {
    // init mb_intra_vec
    mb_intra_vec = vector<vector<string>>(240, vector<string>(320, "0"));
    pic_num += 1;
    picture_start_code = read_bits(32);
    temporal_reference = read_bits(10);
    picture_coding_type = read_bits(3);
    vbv_delay = read_bits(16);

    // debug
//    cout << "==PICTURE_START_CODE(";
//    cout << dec << pic_num;
//    cout << ")== -> ";
//    print_hex(picture_start_code);
//    cout << "    temporal_reference: ";
//    print_hex(temporal_reference);
//    cout << "    picture_coding_type: ";
//    print_hex(picture_coding_type);
//    cout << "    vbv_delay: ";
//    print_dec(vbv_delay);

    if (picture_coding_type == 2 || picture_coding_type == 3) {
        full_pel_forward_vector = read_bits(1);
        forward_f_code = read_bits(3);
        // decode forward_f_code
        forward_r_size = forward_f_code - 1;
        forward_f = 1 << forward_r_size;
        // debug
//        cout << "    full_pel_forward_vector: ";
//        print_dec(full_pel_forward_vector);
//        cout << "    forward_f_code: ";
//        print_dec(forward_f_code);
    }
    if (picture_coding_type == 3) {
        full_pel_backward_vector = read_bits(1);
        backward_f_code = read_bits(3);
        // decode forward_f_code
        backward_r_size = backward_f_code - 1;
        backward_f = 1 << backward_r_size;
    }
    while (nextbits(1) == 1) {
        extra_bit_picture = read_bits(1);
        extra_information_picture = read_bits(8);
    }
    extra_bit_picture = read_bits(1);
    
    // debug
//    cout << "    extra_bit_picture: ";
//    print_hex(extra_bit_picture);    
    
    if (picture_coding_type != 3) {
        pel_future_R = pel_past_R;
        pel_future_G = pel_past_G;
        pel_future_B = pel_past_B;
    }

    mb_address = 0;
    next_start_code();
    if (is_next_slice_code()) {
        // Slice start code '00000101' to '000001AF'
        do {
            slice();
        } while (is_next_slice_code());
    }
    // Reconstruct I-frame
    if (picture_coding_type == 3) {
        recon_pic();
        ycbcr2rgb(false, true);  // to_pel_past, to_output
    } else {
        // update rgb data to pel_past_R, G, B
        recon_pic();
        ycbcr2rgb(true, false);  // to_pel_past, to_output
        if (pic_num > 1) {
//            img_queue.push_back(pel_future_R);
//            img_queue.push_back(pel_future_G);
//            img_queue.push_back(pel_future_B);
            // push to buffer
            rgb2cvmat(pel_future_R, pel_future_G, pel_future_B);        
        }
    }
//    if (img_queue.size() > 0) {
//        output_img();
//    }
    if (imageQueue.size() > 0) {
        imshow("test", imageQueue.at(0));
        imageQueue.pop_front();
        waitKey(30);
    }
}

void Decoder::slice() {
    // init dct dc past
    dct_dc_y_past = 1024;
    dct_dc_cb_past = 1024;
    dct_dc_cr_past = 1024;
    // init mb address
    mb_address = (slice_vertical_position - 1) * mb_width - 1;
    // init past intra address
    past_intra_address = -2;
    // init recon reight & down prev for forward & backward
    recon_right_for_prev = 0;
    recon_down_for_prev = 0;
    recon_right_bac_prev = 0;
    recon_down_bac_prev = 0;

    slice_start_code = read_bits(32);
    quantizer_scale = read_bits(5);

    uint8_t nbs = nextbits(1);
    if (nbs == 0x1) {
        // debug
//        cout << "[ TODO ] : extra_bit_slice & extra_information_slice ..." << endl;
    }
    extra_bit_slice = read_bits(1);

    // debug
//    cout << "==SLICE_START_CODE== -> ";    
//    print_hex(slice_start_code);
//    cout << "    quantizer_scale: ";
//    print_hex(quantizer_scale);
//    cout << "    extra_bit_slice: ";
//    print_hex(extra_bit_slice);

    // macroblock();
    do {
        macroblock();
    } while (nextbits(23) != 0);
    next_start_code();
}

void Decoder::macroblock() {
    // init pattern_code
    pattern_code = {0, 0, 0, 0, 0, 0};
    // init cbp
    cbp = 0;

    // read macroblock stuffing & escape
    uint16_t nbs11 = nextbits(11);
    while (nbs11 == 0xF) {
        mb_stuffing = read_bits(11);
    }
//    nbs11 = nextbits(11);
//    if (nbs11 == 0x8) {
//        mb_escape = read_bits(11);
//    }

    // read macroblock increment number
    //      search 1-11 bits > get int
    int mb_inc = 0;
    int inc_acc = 0;
    do {
        // debug
//        string s = bitset<32>(nextbits(32)).to_string();
//        cout << s << endl;
        mb_inc = get_mb_address_map();
        if (mb_inc == 101) {
            // mb_address_increment = 33;
            inc_acc += 33;
//            cout << "          macroblcok_escape" << endl;
        } else {
            // mb_address_increment = mb_inc;
            inc_acc += mb_inc;
        }
    } while (mb_inc == 101);

    // reset dct_dc_past to 1024 when skipped & mb_intra == "0"
    if ((inc_acc > 1) || (mb_intra == "0")) {
        dct_dc_y_past = 1024;
        dct_dc_cb_past = 1024;
        dct_dc_cr_past = 1024;
    }

    // process skipped macroblocks
    for (int i=1; i<inc_acc; i++) {
        // update mb_address & (mb_row, mb_col)
        mb_address += 1;
        mb_row = mb_address / mb_width;
        mb_col = mb_address % mb_width;
        if (picture_coding_type == 2) {
            recon_right_for = 0;
            recon_down_for = 0;
            recon_right_for_prev = 0;
            recon_down_for_prev = 0;
            decode_mv();
        } else if (picture_coding_type == 3) {          
            decode_mv();
        }
    }
    // update mb_address & (mb_row, mb_col)
    mb_address += 1;
    mb_row = mb_address / mb_width;
    mb_col = mb_address % mb_width;

    // read 1-6 bits > get 5 bits string
    mb_type = get_mb_type_map();
    mb_quant = mb_type.substr(0, 1);
    mb_motion_forward = mb_type.substr(1, 1);
    mb_motion_backward = mb_type.substr(2, 1);
    mb_pattern = mb_type.substr(3, 1);
    mb_intra = mb_type.substr(4, 1);

    // debug
//    cout << "    ==MACROBLOCK(";
//    cout << dec << (mb_address + 1);
//    cout << ")==" << endl;
//    cout << "        macroblock_address_increment: ";
//    cout << mb_address_increment << endl;
//    cout << "        macroblock_type: ";
//    cout << stoi(mb_type, nullptr, 2) << endl;
//    cout << "            macroblcok_quant: ";
//    cout << mb_quant << endl;
//    cout << "            macroblcok_motion_forward: ";
//    cout << mb_motion_forward << endl;
//    cout << "            macroblcok_motion_backward: ";    
//    cout << mb_motion_backward << endl;
//    cout << "            macroblcok_pattern: ";
//    cout << mb_pattern << endl;
//    cout << "            macroblcok_intra: ";    
//    cout << mb_intra << endl;

    // init recon_right_for & recon_down_for for B-frame
    if ((picture_coding_type == 3) && (mb_intra == "1")) {
        recon_right_for_prev = 0;
        recon_down_for_prev = 0;
        recon_right_bac_prev = 0;
        recon_down_bac_prev = 0;
    }

    // init for motion vector
    recon_right_for = 0;
    recon_down_for = 0;

    if (mb_quant == "1") {
        quantizer_scale = read_bits(5);
        // debug
//        cout << "        quantier_scale: ";
//        print_dec(quantizer_scale);
    }
    if (mb_motion_forward == "1") {
        // horizontal
        motion_horizontal_forward_code = get_motion_vector_map();
        // debug
//        cout << "        motion_horizontal_forward_code: ";
//        cout << motion_horizontal_forward_code << endl;
        if (((forward_f) != 1) && (motion_horizontal_forward_code != 0)) {
            motion_horizontal_forward_r = read_bits(forward_r_size);
            // debug
//            cout << "        motion_horizontal_forward_r: ";
//            print_dec(motion_horizontal_forward_r);
        }
        // vertical
        motion_vertical_forward_code = get_motion_vector_map();
        // debug
//        cout << "        motion_vertical_forward_code: ";
//        cout << motion_vertical_forward_code << endl;
        if (((forward_f) != 1) && (motion_vertical_forward_code != 0)) {
            motion_vertical_forward_r = read_bits(forward_r_size);
            // debug
//            cout << "        motion_vertical_forward_r: ";
//            print_dec(motion_vertical_forward_r);
        }
        // motion vectors
        cal_motion_vector_p();    
    } else if (picture_coding_type == 2) {
        recon_right_for = 0;
        recon_down_for = 0;
        recon_right_for_prev = 0;
        recon_down_for_prev = 0;
        mb_motion_forward = "1";
    }

    // init back motion vector
    recon_right_bac = 0;
    recon_down_bac = 0;
    if (mb_motion_backward == "1") {
        // horizontal
        motion_horizontal_backward_code = get_motion_vector_map();
        // debug
//        cout << "        motion_horizontal_backward_code: ";
//        cout << motion_horizontal_backward_code << endl;        
        if (((backward_f) != 1) && (motion_horizontal_backward_code != 0)) {
            motion_horizontal_backward_r = read_bits(backward_r_size);
            // debug
//            cout << "        motion_horizontal_backward_r: ";
//            print_dec(motion_horizontal_backward_r);
        }
        // vertical
        motion_vertical_backward_code = get_motion_vector_map();
        // debug
//        cout << "        motion_vertical_backward_code: ";
//        cout << motion_vertical_backward_code << endl;        
        if (((backward_f) != 1) && (motion_vertical_backward_code != 0)) {
            motion_vertical_backward_r = read_bits(backward_r_size);
            // debug
//            cout << "        motion_vertical_backward_r: ";
//            print_dec(motion_vertical_backward_r);
        }
        // motion vectors
        cal_motion_vector_b();
    }

    // init dct dc past
    if (mb_intra == "0") {
        decode_mv();
        // debug
//        cout << "        ForMV(";
//        cout << recon_right_for << ",";
//        cout << recon_down_for << ")" << endl;   
    } else {
        for (int i=0; i<16; i++) {
            for (int j=0; j<16; j++) {
                int pel_r = (mb_row * 16) + i;
                int pel_c = (mb_col * 16) + j;
                mb_intra_vec.at(pel_r).at(pel_c) = mb_intra;
            }
        }
    }
    // update cbp
    if (mb_pattern == "1") {
        coded_block_pattern();
    }
    update_pattern_code(pattern_code);
    // blocks loop
    for (int i=0; i<6; i++) {
        block(i);
    }
    // update past mb address
    if (mb_intra == "1") {
        past_intra_address = mb_address;
    }
    // end_of_macroblock
    if (picture_coding_type == 4) {
        uint8_t _ = read_bits(1);
    }
}

void Decoder::block(int i) {
    block_i = i;
    // init dct zz i
    dct_zz_i = 0;
    // init dct_zz
    dct_zz = vector<int>(64, 0);
    // main loop
    if (pattern_code[i] == 1) {
        // debug
        // cout << "        ==BLOCK(" << i << ")==" << endl;
        if (mb_intra == "1") {
            if (i<4) {
                // dct_dc_size_luminance = get_dct_dc_size_lum_map();
                dct_dc_size_luminance = get_dct_dc_size_lum_map_s();
                if(dct_dc_size_luminance != 0) {
                    dct_dc_differential = read_bits(dct_dc_size_luminance);   
                    if (dct_dc_differential & (1 << (dct_dc_size_luminance-1))) {
                        dct_zz.at(0) = dct_dc_differential;
                    } else {
                        dct_zz.at(0) = (-1 << (dct_dc_size_luminance)) | (dct_dc_differential+1);
                    }
                    // debug
                    // cout << "            dct_dc_size_luminance: ";
                    // cout << dct_dc_size_luminance << endl;
                    // cout << "            dct_dc_differential_luminance: ";
                    // print_dec(dct_dc_differential);
                } else {
                    dct_zz.at(0) = 0;
                    // debug
                    // cout << "            dct_dc_size_luminance: ";
                    // cout << dct_dc_size_luminance << endl;                    
                }
            } else {
                dct_dc_size_chrominance = get_dct_dc_size_chr_map();
                // dct_dc_size_chrominance = get_dct_dc_size_chr_map_s();
                if(dct_dc_size_chrominance !=0) {
                    dct_dc_differential = read_bits(dct_dc_size_chrominance);
                    if (dct_dc_differential & (1 << (dct_dc_size_chrominance-1))) {
                        dct_zz.at(0) = dct_dc_differential;
                    } else {
                        dct_zz.at(0) = (-1 << (dct_dc_size_chrominance)) | (dct_dc_differential+1) ;
                    }
                    // debug
                    // cout << "            dct_dc_size_chrominance: ";
                    // cout << dct_dc_size_chrominance << endl;
                    // cout << "            dct_dc_differential_luminance: ";     
                    // print_dec(dct_dc_differential);
                } else {
                    dct_zz.at(0) = 0;
                    // debug
                    // cout << "            dct_dc_size_chrominance: ";
                    // cout << dct_dc_size_chrominance << endl;          
                }
            }
        } else {
            // dct_coeff_first();
            dct_coeff_first_s();
        }
        if (picture_coding_type != 4) {
            // If mb_intra == 1 then i shall be set to zero before the first dct_coeff_next. 
            if (mb_intra == "1") {
                dct_zz_i = 0;
            }
            while (nextbits(2) != 2) {
                // dct_coeff_next();
                dct_coeff_next_s();
            }
            end_of_block = read_bits(2);
            // debug
            if (end_of_block == 2) {
                // cout << "            EOB" << endl;
            }
            // Reconstruct dct_recon
            reconstruct_dct(i);
            // IDCT
            idct();
        }
    }
}

// Utils
int Decoder::sign(int num) {
    if (num > 0) {
        return 1;
    } else if (num == 0) {
        return 0;
    } else {
        return -1;
    }
}

void Decoder::load_intra_quant() {
    for (int i=0; i<8; i++) {
        for (int j=0; j<8; j++) {
            intra_quant.at(i).at(j) = read_bits(8);
        }
    }
}

void Decoder::load_non_intra_quant() {
    for (int i=0; i<8; i++) {
        for (int j=0; j<8; j++) {
            non_intra_quant.at(i).at(j) = read_bits(8);
        }
    }
}

void Decoder::print_hex(unsigned int code) {
    cout << hex << code << endl;
}

void Decoder::print_dec(unsigned int code) {
    cout << dec << code << endl;
}

// ---> check start & end code
void Decoder::next_start_code() {
    uint32_t next_buf = 0;
    for (int i=0; i<4; i++) {
        next_buf = (next_buf << 8) + (abs_buf.at(i));
    }
    if ((next_buf >> 8) == 0x1) {
        zero_byte_flag = true;
        uint32_t tmp_buf = next_buf << 24;
        zero_byte = tmp_buf >> 24;
    }
}

bool Decoder::is_next_start_code(int code) {
    if ((zero_byte_flag == true) && (zero_byte == code)) {
        buf = 0;
        buf_cursor = 0;
        zero_byte_flag = false;
        return true;
    } else {
        return false;
    }
}

bool Decoder::is_next_slice_code() {
    if (zero_byte_flag == true) {
        if ((zero_byte >= 0x1) && (zero_byte <= 0xAF)) {
            buf = 0;
            buf_cursor = 0;
            zero_byte_flag = false;
            slice_vertical_position = zero_byte;
            return true;
        } else {
            return false;
        }
    } else {
        return false;
    }
}

uint32_t Decoder::nextbits(int num) {
    uint32_t bit32  = 0;
    
//    cout << "----- nextbits -----" << endl;
//    string s = bitset<32>(bit32).to_string();
//    cout << s << endl;
    
    // Head
    int i = 0;
    int total_rem = num;
    int head_rem = 8 - buf_cursor;
    if (num <= head_rem) {
        // update total_rem
        total_rem = 0;
        if (buf_cursor == 0) {
            // update buffer
            uint8_t tmp_buf = abs_buf.at(i);
            i += 1;
            bit32 = (bit32 << num) + (tmp_buf >> (8 - num));
        } else {
            uint8_t tmp_buf = buf << buf_cursor;
            bit32 = (bit32 << num) + (tmp_buf >> (8 - (buf_cursor + num) + buf_cursor));
        }
    } else {
        if (buf_cursor == 0) {
            // update total_rem
            total_rem -= 8;                    
            // read 8-bit * 1
            bit32 = (bit32 << 8) + abs_buf.at(i);
            i += 1;
        } else {
            // update total_rem
            total_rem -= head_rem;
            // read all buf in
            uint8_t tmp_buf = buf << buf_cursor;
            bit32 = (bit32 << head_rem) + (tmp_buf >> buf_cursor);
        }
    }
    
//    string s2 = bitset<32>(bit32).to_string();
//    cout << s2 << endl;    
    
    // Middle
    int quotient = total_rem / 8;
    
    for (int j=0; j<quotient; i++, j++) {
        // update total_rem
        total_rem -= 8;
        // read 8-bit * N
        bit32 = (bit32 << 8) + abs_buf.at(i);        
    }

//    string s3 = bitset<32>(bit32).to_string();
//    cout << s3 << endl;        
    
    // Last
    if (total_rem > 0) {
        // update buffer
        uint8_t tmp_buf = abs_buf.at(i);
        bit32 = (bit32 << total_rem) + (tmp_buf >> (8 - total_rem));
    }

//    string s4 = bitset<32>(bit32).to_string();
//    cout << s4 << endl;        
    
    return bit32;  
}

uint32_t Decoder::read_bits(int num) {
    uint32_t bit32  = 0;
    
//    string s = bitset<32>(bit32).to_string();
//    cout << s << endl;
//    string b = bitset<8>(buf).to_string();
//    cout << b << endl;    
    
    int new_cur_pos = get_cur_pos(buf_cursor, num);
    // Head
    int total_rem = num;
    int head_rem = 8 - buf_cursor;
    if (num <= head_rem) {
        // update total_rem
        total_rem = 0;        
        if (buf_cursor == 0) {
            // update buffer
            buf = abs_buf.at(0);
            bit32 = (bit32 << num) + (buf >> (8 - num));
            abs_buf.erase(abs_buf.begin(), abs_buf.begin()+1);
        } else {
            uint8_t tmp_buf = buf << buf_cursor;
            bit32 = (bit32 << num) + (tmp_buf >> (8 - (buf_cursor + num) + buf_cursor));
        }
    } else {
        if (buf_cursor == 0) {
            // update total_rem
            total_rem -= 8;                    
            // read 8-bit * 1
            bit32 = (bit32 << 8) + abs_buf.at(0);
            abs_buf.erase(abs_buf.begin(), abs_buf.begin()+1); 
        } else {
            // update total_rem
            total_rem -= head_rem;
            // read all buf in
            uint8_t tmp_buf = buf << buf_cursor;
            bit32 = (bit32 << head_rem) + (tmp_buf >> buf_cursor);
        }
    }
    
//    string s1 = bitset<32>(bit32).to_string();
//    cout << s1 << endl;
//    string b1 = bitset<8>(buf).to_string();
//    cout << b1 << endl;
    
    // Middle
    int quotient = total_rem / 8;
    for (int i=0; i<quotient; i++) {
        // update total_rem
        total_rem -= 8;
        // read 8-bit * N
        bit32 = (bit32 << 8) + abs_buf.at(0);
        abs_buf.erase(abs_buf.begin(), abs_buf.begin()+1);
    }
    
//    string s2 = bitset<32>(bit32).to_string();
//    cout << s2 << endl;
//    string b2 = bitset<8>(buf).to_string();
//    cout << b2 << endl;    
    
    // Last
    if (total_rem > 0) {
        // update buffer
        buf = abs_buf.at(0);
        bit32 = (bit32 << total_rem) + (buf >> (8 - total_rem));
        abs_buf.erase(abs_buf.begin(), abs_buf.begin()+1);
    }
    
//    string s3 = bitset<32>(bit32).to_string();
//    cout << s3 << endl;
//    string b3 = bitset<8>(buf).to_string();
//    cout << b3 << endl;   
//    cout << endl;
    
    // update cursor position
    buf_cursor = new_cur_pos;
    // modify buffer
    if (buf_cursor == 0) {
        buf = 0;
    }
    return bit32;    
}

// ----> other layers
int Decoder::get_cur_pos(int cur_pos, int num) {
    int add_num = num % 8;
    int remain_num = (((cur_pos + 1) + add_num) % 8 - 1);
    if (remain_num == -1) {
        remain_num = 7;
    }
    return remain_num;
}

void Decoder::update_pattern_code(vector<int> & pattern_code) {
    // update pattern code
    for (int i=0; i<6; i++) {
        pattern_code.at(i) = 0;
        if (cbp & (1 << (5-i))) {
            pattern_code.at(i) = 1;
        }
        if (mb_intra == "1") {
            pattern_code.at(i) = 1;
        }
    }
    // debug
    if (mb_pattern == "1") {
        uint8_t _tmp = 0;
        for (int i=0; i<6; i++) {
            if (pattern_code.at(i) == 0) {
                _tmp = (_tmp << 1) + 0;
            } else if (pattern_code.at(i) == 1) {
                _tmp = (_tmp << 1) + 1;
            }
        }
//        cout << "        coded_block_pattern: ";
//        print_dec(_tmp);
    }
}

// Get Mapping Value
int Decoder::get_mb_address_map() {
    // debug
//    cout << "[ Info ] : in `get_mb_address_map()` function ..." << endl;
    uint16_t nbs11 = nextbits(11);
    uint16_t tmp_nbs = 0;
    for (int i=1; i<12; i++) {
        string s = bitset<11>(nbs11).to_string();
        s = s.substr(0, i);
        // debug
//        cout << "    --> searching i = " << i;
//        cout << " (string : " << s << ")" << endl;
        map<string, int>::iterator iter;
        iter = mb_address_map.find(s);
        if (iter !=mb_address_map.end()) {
            // debug
//            cout << "        --> get result : ";
//            cout << iter->second << endl;
            uint8_t _ = read_bits(i);
            return iter->second;
        }
    }
}

string Decoder::get_mb_type_map() {
    // debug
//    cout << "[ Info ] : in `get_mb_type_map()` function ..." << endl;
    uint8_t nbs6 = nextbits(6);
    uint8_t tmp_nbs = 0;
    for (int i=1; i<7; i++) {
        string s = bitset<6>(nbs6).to_string();
        s = s.substr(0, i);
        // debug
//        cout << "    --> searching i = " << i;
//        cout << " (string : " << s << ")" << endl;
        map<string, string>::iterator iter;
        if (picture_coding_type == 1) {
            // i-frame
            iter = mb_type_i_map.find(s);
            if (iter !=mb_type_i_map.end()) {
                    // debug
//                    cout << "        --> get result : ";
//                    cout << iter->second << endl;
                    uint8_t _ = read_bits(i);
                    return iter->second;
            }         
        } else if (picture_coding_type == 2) {
            // p-frame
            iter = mb_type_p_map.find(s);
            if (iter !=mb_type_p_map.end()) {
                    // debug
//                    cout << "        --> get result : ";
//                    cout << iter->second << endl;
                    uint8_t _ = read_bits(i);
                    return iter->second;
            }          
        } else if (picture_coding_type == 3) {
            // b-frame
            iter = mb_type_b_map.find(s);
            if (iter !=mb_type_b_map.end()) {
                    // debug
//                    cout << "        --> get result : ";
//                    cout << iter->second << endl;
                    uint8_t _ = read_bits(i);
                    return iter->second;
            }      
        }
    }
}

int Decoder::get_dct_dc_size_lum_map() {
    // debug
//    cout << "            [ Info ] : in `dct_dc_size_lum_map()` function ..." << endl;
    uint8_t nbs7 = nextbits(7);
    for (int i=2; i<8; i++) {
        string s = bitset<7>(nbs7).to_string();
        s = s.substr(0, i);
        // debug
//        cout << "                --> searching i = " << i;
//        cout << " (string : " << s << ")" << endl;
        map<string, int>::iterator iter;
        iter = dct_dc_size_lum_map.find(s);
        if (iter !=dct_dc_size_lum_map.end()) {
            // debug
//            cout << "                    --> get result : ";
//            cout << iter->second << endl;
            uint8_t _ = read_bits(i);
            return iter->second;
        }
    }
}

int Decoder::get_dct_dc_size_lum_map_s() {
    uint16_t nbs16 = 0;
    uint8_t second_nbs8 = 0;
    for (int i=2; i<8; i++) {
        second_nbs8 = nextbits(i);
        nbs16 = i;
        nbs16 = (nbs16 << 8) + second_nbs8;    
        // Lookup table via switch optimization
        uint8_t _ = 0;
        switch (nbs16) {
            case 0x200:
                _ = read_bits(i);
                i = 8;
                return 1;
            case 0x201:
                _ = read_bits(i);
                i = 8;
                return 2;
            case 0x304:
                _ = read_bits(i);
                i = 8;
                return 0;
            case 0x305:
                _ = read_bits(i);
                i = 8;
                return 3;
            case 0x306:
                _ = read_bits(i);
                i = 8;
                return 4;
            case 0x40e:
                _ = read_bits(i);
                i = 8;
                return 5;
            case 0x51e:
                _ = read_bits(i);
                i = 8;
                return 6;
            case 0x63e:
                _ = read_bits(i);
                i = 8;
                return 7;
            case 0x77e:
                _ = read_bits(i);
                i = 8;
                return 8;
        }
    }
}

int Decoder::get_dct_dc_size_chr_map() {
    // debug
//    cout << "            [ Info ] : in `get_dct_dc_size_chr_map()` function ..." << endl;
    uint8_t nbs8 = nextbits(8);
    for (int i=2; i<9; i++) {
        string s = bitset<8>(nbs8).to_string();
        s = s.substr(0, i);
        // debug
//        cout << "                --> searching i = " << i;
//        cout << " (string : " << s << ")" << endl;
        map<string, int>::iterator iter;
        iter = dct_dc_size_chr_map.find(s);
        if (iter !=dct_dc_size_chr_map.end()) {
            // debug
//            cout << "                    --> get result : ";
//            cout << iter->second << endl;
            uint8_t _ = read_bits(i);
            return iter->second;
        }
    }
}

int Decoder::get_dct_dc_size_chr_map_s() {
    uint16_t nbs16 = 0;
    uint8_t second_nbs8 = 0;
    for (int i=2; i<8; i++) {
        second_nbs8 = nextbits(i);
        nbs16 = i;
        nbs16 = (nbs16 << 8) + second_nbs8;    
        // Lookup table via switch optimization
        uint8_t _ = 0;
        switch (nbs16) {
            case 0x200:
                _ = read_bits(i);
                i = 8;
                return 0;
            case 0x201:
                _ = read_bits(i);
                i = 8;
                return 1;
            case 0x202:
                _ = read_bits(i);
                i = 8;
                return 2;
            case 0x306:
                _ = read_bits(i);
                i = 8;
                return 3;
            case 0x40e:
                _ = read_bits(i);
                i = 8;
                return 4;
            case 0x51e:
                _ = read_bits(i);
                i = 8;
                return 5;
            case 0x63e:
                _ = read_bits(i);
                i = 8;
                return 6;
            case 0x77e:
                _ = read_bits(i);
                i = 8;
                return 7;
            case 0x8fe:
                _ = read_bits(i);
                i = 8;
                return 8;
        }
    }
}

int Decoder::get_motion_vector_map() {
    uint16_t nbs11 = nextbits(11);
    for (int i=1; i<12; i++) {
        string s = bitset<11>(nbs11).to_string();
        s = s.substr(0, i);
        // debug
//        cout << "                --> searching i = " << i;
//        cout << " (string : " << s << ")" << endl;
        map<string, int>::iterator iter;
        iter = motion_vector_map.find(s);
        if (iter !=motion_vector_map.end()) {
            // debug
//            cout << "                    --> get result : ";
//            cout << iter->second << endl;
            uint8_t _ = read_bits(i);
            return iter->second;
        }
    }
}

// Reconstruct I-frame
void Decoder::coded_block_pattern() {
    uint16_t nbs9 = nextbits(9);
    for (int i=3; i<10; i++) {
        string s = bitset<9>(nbs9).to_string();
        s = s.substr(0, i);
        map<string, int>::iterator iter;
        iter = mb_pattern_map.find(s);
        if (iter !=mb_pattern_map.end()) {
            uint8_t _ = read_bits(i);
            cbp = iter->second;
            break;
        }
    }
}

void Decoder::dct_coeff_first() {
    int run = 0;
    int level = 0;
    // Get run and level from mapping
    // debug
    cout << "            [ Info ] : in `dct_coeff_first()` function ..." << endl;    
    uint32_t nbs28 = nextbits(28);
    for (int i=1; i<29; i++) {
        string s = bitset<28>(nbs28).to_string();
        s = s.substr(0, i);
        // debug
        cout << "                --> searching i = " << i;
        cout << " (string : " << s << ")" << endl;
        map<string, vector<int>>::iterator iter;
        iter = dct_coeff_first_map.find(s);
        if (iter !=dct_coeff_first_map.end()) {
            // debug
            cout << "                    --> get run : ";
            cout << iter->second.at(0) << endl;
            cout << "                    --> get level : ";
            cout << iter->second.at(1) << endl;
            uint8_t _ = read_bits(i);
            run = iter->second.at(0);
            level = iter->second.at(1);
            // normal case
            if (run != -1) {
                // Get "s"
                // debug
                cout << "            dct_coeff_first: ";
                dct_zz_i = run;
                int s = read_bits(1);
                if (s == 0) {
                    dct_zz.at(dct_zz_i) = level;
                    // debug
                    cout << "Run: " << run;
                    cout << " Level: ";
                    cout << dec << level << endl;
                } else if (s == 1) {
                    int neg_level = - level;
                    dct_zz.at(dct_zz_i) = neg_level;
                    // debug
                    cout << "Run: " << run;
                    cout << " Level: ";
                    cout << dec << neg_level << endl;
                }
                // debug
                cout << "                    --> get s : ";
//                print_hex(s);            
                break;
            } else {
                // escape case
                run = get_escape_run();
                level = get_escape_level();
                dct_zz_i = run;
                dct_zz.at(dct_zz_i) = level;
                // debug
                cout << "            dct_coeff_first: ";
                cout << "ESCAPE Run: " << run;
                cout << " Level: ";
                cout << dec << level << endl;
            }
        }
    }    
}

void Decoder::dct_coeff_first_s() {
    uint64_t nbs64 = 0;
    uint32_t second_nbs32 = 0;
    for (int i=1; i<29; i++) {
        second_nbs32 = nextbits(i);
        nbs64 = i;
        nbs64 = (nbs64 << 32) + second_nbs32;
        // Lookup table via switch optimization
        uint8_t _ = 0;
        switch (nbs64) {
            case 0x1000000010:
                _ = read_bits(i);
                fill_dct_zz_first(1, 18);
                i = 29;
                break;
            case 0x1000000011:
                _ = read_bits(i);
                fill_dct_zz_first(1, 17);
                i = 29;
                break;
            case 0x1000000012:
                _ = read_bits(i);
                fill_dct_zz_first(1, 16);
                i = 29;
                break;
            case 0x1000000013:
                _ = read_bits(i);
                fill_dct_zz_first(1, 15);
                i = 29;
                break;
            case 0x1000000014:
                _ = read_bits(i);
                fill_dct_zz_first(6, 3);
                i = 29;
                break;
            case 0x1000000015:
                _ = read_bits(i);
                fill_dct_zz_first(16, 2);
                i = 29;
                break;
            case 0x1000000016:
                _ = read_bits(i);
                fill_dct_zz_first(15, 2);
                i = 29;
                break;
            case 0x1000000017:
                _ = read_bits(i);
                fill_dct_zz_first(14, 2);
                i = 29;
                break;
            case 0x1000000018:
                _ = read_bits(i);
                fill_dct_zz_first(13, 2);
                i = 29;
                break;
            case 0x1000000019:
                _ = read_bits(i);
                fill_dct_zz_first(12, 2);
                i = 29;
                break;
            case 0x100000001a:
                _ = read_bits(i);
                fill_dct_zz_first(11, 2);
                i = 29;
                break;
            case 0x100000001b:
                _ = read_bits(i);
                fill_dct_zz_first(31, 1);
                i = 29;
                break;
            case 0x100000001c:
                _ = read_bits(i);
                fill_dct_zz_first(30, 1);
                i = 29;
                break;
            case 0x100000001d:
                _ = read_bits(i);
                fill_dct_zz_first(29, 1);
                i = 29;
                break;
            case 0x100000001e:
                _ = read_bits(i);
                fill_dct_zz_first(28, 1);
                i = 29;
                break;
            case 0x100000001f:
                _ = read_bits(i);
                fill_dct_zz_first(27, 1);
                i = 29;
                break;
            case 0xf00000010:
                _ = read_bits(i);
                fill_dct_zz_first(0, 40);
                i = 29;
                break;
            case 0xf00000011:
                _ = read_bits(i);
                fill_dct_zz_first(0, 39);
                i = 29;
                break;
            case 0xf00000012:
                _ = read_bits(i);
                fill_dct_zz_first(0, 38);
                i = 29;
                break;
            case 0xf00000013:
                _ = read_bits(i);
                fill_dct_zz_first(0, 37);
                i = 29;
                break;
            case 0xf00000014:
                _ = read_bits(i);
                fill_dct_zz_first(0, 36);
                i = 29;
                break;
            case 0xf00000015:
                _ = read_bits(i);
                fill_dct_zz_first(0, 35);
                i = 29;
                break;
            case 0xf00000016:
                _ = read_bits(i);
                fill_dct_zz_first(0, 34);
                i = 29;
                break;
            case 0xf00000017:
                _ = read_bits(i);
                fill_dct_zz_first(0, 33);
                i = 29;
                break;
            case 0xf00000018:
                _ = read_bits(i);
                fill_dct_zz_first(0, 32);
                i = 29;
                break;
            case 0xf00000019:
                _ = read_bits(i);
                fill_dct_zz_first(1, 14);
                i = 29;
                break;
            case 0xf0000001a:
                _ = read_bits(i);
                fill_dct_zz_first(1, 13);
                i = 29;
                break;
            case 0xf0000001b:
                _ = read_bits(i);
                fill_dct_zz_first(1, 12);
                i = 29;
                break;
            case 0xf0000001c:
                _ = read_bits(i);
                fill_dct_zz_first(1, 11);
                i = 29;
                break;
            case 0xf0000001d:
                _ = read_bits(i);
                fill_dct_zz_first(1, 10);
                i = 29;
                break;
            case 0xf0000001e:
                _ = read_bits(i);
                fill_dct_zz_first(1, 9);
                i = 29;
                break;
            case 0xf0000001f:
                _ = read_bits(i);
                fill_dct_zz_first(1, 8);
                i = 29;
                break;
            case 0xe00000010:
                _ = read_bits(i);
                fill_dct_zz_first(0, 31);
                i = 29;
                break;
            case 0xe00000011:
                _ = read_bits(i);
                fill_dct_zz_first(0, 30);
                i = 29;
                break;
            case 0xe00000012:
                _ = read_bits(i);
                fill_dct_zz_first(0, 29);
                i = 29;
                break;
            case 0xe00000013:
                _ = read_bits(i);
                fill_dct_zz_first(0, 28);
                i = 29;
                break;
            case 0xe00000014:
                _ = read_bits(i);
                fill_dct_zz_first(0, 27);
                i = 29;
                break;
            case 0xe00000015:
                _ = read_bits(i);
                fill_dct_zz_first(0, 26);
                i = 29;
                break;
            case 0xe00000016:
                _ = read_bits(i);
                fill_dct_zz_first(0, 25);
                i = 29;
                break;
            case 0xe00000017:
                _ = read_bits(i);
                fill_dct_zz_first(0, 24);
                i = 29;
                break;
            case 0xe00000018:
                _ = read_bits(i);
                fill_dct_zz_first(0, 23);
                i = 29;
                break;
            case 0xe00000019:
                _ = read_bits(i);
                fill_dct_zz_first(0, 22);
                i = 29;
                break;
            case 0xe0000001a:
                _ = read_bits(i);
                fill_dct_zz_first(0, 21);
                i = 29;
                break;
            case 0xe0000001b:
                _ = read_bits(i);
                fill_dct_zz_first(0, 20);
                i = 29;
                break;
            case 0xe0000001c:
                _ = read_bits(i);
                fill_dct_zz_first(0, 19);
                i = 29;
                break;
            case 0xe0000001d:
                _ = read_bits(i);
                fill_dct_zz_first(0, 18);
                i = 29;
                break;
            case 0xe0000001e:
                _ = read_bits(i);
                fill_dct_zz_first(0, 17);
                i = 29;
                break;
            case 0xe0000001f:
                _ = read_bits(i);
                fill_dct_zz_first(0, 16);
                i = 29;
                break;
            case 0xd00000010:
                _ = read_bits(i);
                fill_dct_zz_first(10, 2);
                i = 29;
                break;
            case 0xd00000011:
                _ = read_bits(i);
                fill_dct_zz_first(9, 2);
                i = 29;
                break;
            case 0xd00000012:
                _ = read_bits(i);
                fill_dct_zz_first(5, 3);
                i = 29;
                break;
            case 0xd00000013:
                _ = read_bits(i);
                fill_dct_zz_first(3, 4);
                i = 29;
                break;
            case 0xd00000014:
                _ = read_bits(i);
                fill_dct_zz_first(2, 5);
                i = 29;
                break;
            case 0xd00000015:
                _ = read_bits(i);
                fill_dct_zz_first(1, 7);
                i = 29;
                break;
            case 0xd00000016:
                _ = read_bits(i);
                fill_dct_zz_first(1, 6);
                i = 29;
                break;
            case 0xd00000017:
                _ = read_bits(i);
                fill_dct_zz_first(0, 15);
                i = 29;
                break;
            case 0xd00000018:
                _ = read_bits(i);
                fill_dct_zz_first(0, 14);
                i = 29;
                break;
            case 0xd00000019:
                _ = read_bits(i);
                fill_dct_zz_first(0, 13);
                i = 29;
                break;
            case 0xd0000001a:
                _ = read_bits(i);
                fill_dct_zz_first(0, 12);
                i = 29;
                break;
            case 0xd0000001b:
                _ = read_bits(i);
                fill_dct_zz_first(26, 1);
                i = 29;
                break;
            case 0xd0000001c:
                _ = read_bits(i);
                fill_dct_zz_first(25, 1);
                i = 29;
                break;
            case 0xd0000001d:
                _ = read_bits(i);
                fill_dct_zz_first(24, 1);
                i = 29;
                break;
            case 0xd0000001e:
                _ = read_bits(i);
                fill_dct_zz_first(23, 1);
                i = 29;
                break;
            case 0xd0000001f:
                _ = read_bits(i);
                fill_dct_zz_first(22, 1);
                i = 29;
                break;
            case 0xc00000010:
                _ = read_bits(i);
                fill_dct_zz_first(0, 11);
                i = 29;
                break;
            case 0xc00000011:
                _ = read_bits(i);
                fill_dct_zz_first(8, 2);
                i = 29;
                break;
            case 0xc00000012:
                _ = read_bits(i);
                fill_dct_zz_first(4, 3);
                i = 29;
                break;
            case 0xc00000013:
                _ = read_bits(i);
                fill_dct_zz_first(0, 10);
                i = 29;
                break;
            case 0xc00000014:
                _ = read_bits(i);
                fill_dct_zz_first(2, 4);
                i = 29;
                break;
            case 0xc00000015:
                _ = read_bits(i);
                fill_dct_zz_first(7, 2);
                i = 29;
                break;
            case 0xc00000016:
                _ = read_bits(i);
                fill_dct_zz_first(21, 1);
                i = 29;
                break;
            case 0xc00000017:
                _ = read_bits(i);
                fill_dct_zz_first(20, 1);
                i = 29;
                break;
            case 0xc00000018:
                _ = read_bits(i);
                fill_dct_zz_first(0, 9);
                i = 29;
                break;
            case 0xc00000019:
                _ = read_bits(i);
                fill_dct_zz_first(19, 1);
                i = 29;
                break;
            case 0xc0000001a:
                _ = read_bits(i);
                fill_dct_zz_first(18, 1);
                i = 29;
                break;
            case 0xc0000001b:
                _ = read_bits(i);
                fill_dct_zz_first(1, 5);
                i = 29;
                break;
            case 0xc0000001c:
                _ = read_bits(i);
                fill_dct_zz_first(3, 3);
                i = 29;
                break;
            case 0xc0000001d:
                _ = read_bits(i);
                fill_dct_zz_first(0, 8);
                i = 29;
                break;
            case 0xc0000001e:
                _ = read_bits(i);
                fill_dct_zz_first(6, 2);
                i = 29;
                break;
            case 0xc0000001f:
                _ = read_bits(i);
                fill_dct_zz_first(17, 1);
                i = 29;
                break;
            case 0xa00000008:
                _ = read_bits(i);
                fill_dct_zz_first(16, 1);
                i = 29;
                break;
            case 0xa00000009:
                _ = read_bits(i);
                fill_dct_zz_first(5, 2);
                i = 29;
                break;
            case 0xa0000000a:
                _ = read_bits(i);
                fill_dct_zz_first(0, 7);
                i = 29;
                break;
            case 0xa0000000b:
                _ = read_bits(i);
                fill_dct_zz_first(2, 3);
                i = 29;
                break;
            case 0xa0000000c:
                _ = read_bits(i);
                fill_dct_zz_first(1, 4);
                i = 29;
                break;
            case 0xa0000000d:
                _ = read_bits(i);
                fill_dct_zz_first(15, 1);
                i = 29;
                break;
            case 0xa0000000e:
                _ = read_bits(i);
                fill_dct_zz_first(14, 1);
                i = 29;
                break;
            case 0xa0000000f:
                _ = read_bits(i);
                fill_dct_zz_first(4, 2);
                i = 29;
                break;
            case 0x600000001:
                _ = read_bits(i);
                fill_dct_zz_first(-1, -1);
                i = 29;
                break;
            case 0x700000004:
                _ = read_bits(i);
                fill_dct_zz_first(2, 2);
                i = 29;
                break;
            case 0x700000005:
                _ = read_bits(i);
                fill_dct_zz_first(9, 1);
                i = 29;
                break;
            case 0x700000006:
                _ = read_bits(i);
                fill_dct_zz_first(0, 4);
                i = 29;
                break;
            case 0x700000007:
                _ = read_bits(i);
                fill_dct_zz_first(8, 1);
                i = 29;
                break;
            case 0x600000004:
                _ = read_bits(i);
                fill_dct_zz_first(7, 1);
                i = 29;
                break;
            case 0x600000005:
                _ = read_bits(i);
                fill_dct_zz_first(6, 1);
                i = 29;
                break;
            case 0x600000006:
                _ = read_bits(i);
                fill_dct_zz_first(1, 2);
                i = 29;
                break;
            case 0x600000007:
                _ = read_bits(i);
                fill_dct_zz_first(5, 1);
                i = 29;
                break;
            case 0x800000020:
                _ = read_bits(i);
                fill_dct_zz_first(13, 1);
                i = 29;
                break;
            case 0x800000021:
                _ = read_bits(i);
                fill_dct_zz_first(0, 6);
                i = 29;
                break;
            case 0x800000022:
                _ = read_bits(i);
                fill_dct_zz_first(12, 1);
                i = 29;
                break;
            case 0x800000023:
                _ = read_bits(i);
                fill_dct_zz_first(11, 1);
                i = 29;
                break;
            case 0x800000024:
                _ = read_bits(i);
                fill_dct_zz_first(3, 2);
                i = 29;
                break;
            case 0x800000025:
                _ = read_bits(i);
                fill_dct_zz_first(1, 3);
                i = 29;
                break;
            case 0x800000026:
                _ = read_bits(i);
                fill_dct_zz_first(0, 5);
                i = 29;
                break;
            case 0x800000027:
                _ = read_bits(i);
                fill_dct_zz_first(10, 1);
                i = 29;
                break;
            case 0x500000005:
                _ = read_bits(i);
                fill_dct_zz_first(0, 3);
                i = 29;
                break;
            case 0x500000006:
                _ = read_bits(i);
                fill_dct_zz_first(4, 1);
                i = 29;
                break;
            case 0x500000007:
                _ = read_bits(i);
                fill_dct_zz_first(3, 1);
                i = 29;
                break;
            case 0x400000004:
                _ = read_bits(i);
                fill_dct_zz_first(0, 2);
                i = 29;
                break;
            case 0x400000005:
                _ = read_bits(i);
                fill_dct_zz_first(2, 1);
                i = 29;
                break;
            case 0x300000003:
                _ = read_bits(i);
                fill_dct_zz_first(1, 1);
                i = 29;
                break;
            case 0x100000001:
                _ = read_bits(i);
                fill_dct_zz_first(0, 1);
                i = 29;
                break;
        }
    }
}

void Decoder::fill_dct_zz_first(int run, int level) {
    // normal case
    if (run != -1) {
        dct_zz_i = run;
        int s = read_bits(1);
        if (s == 0) {
            dct_zz.at(dct_zz_i) = level;
        } else if (s == 1) {
            int neg_level = - level;
            dct_zz.at(dct_zz_i) = neg_level;
        }
    } else {
        // escape case
        run = get_escape_run();
        level = get_escape_level();
        dct_zz_i = run;
        dct_zz.at(dct_zz_i) = level;
    }
}

void Decoder::dct_coeff_next() {
    int run = 0;
    int level = 0;
    // Get run and level from mapping
    // debug
    cout << "            [ Info ] : in `dct_coeff_next()` function ..." << endl;    
    uint32_t nbs28 = nextbits(28);  // 28
    for (int i=2; i<29; i++) {  // 29
        string s = bitset<28>(nbs28).to_string();
        s = s.substr(0, i);
        // debug
        cout << "                --> searching i = " << i;
        cout << " (string : " << s << ")" << endl;
        map<string, vector<int>>::iterator iter;
        iter = dct_coeff_next_map.find(s);
        if (iter !=dct_coeff_next_map.end()) {
            // debug
            cout << "                    --> get run : ";
            cout << dec << iter->second.at(0) << endl;
            cout << "                    --> get level : ";
            cout << dec << iter->second.at(1) << endl;
            uint8_t _ = read_bits(i);
            run = iter->second.at(0);
            level = iter->second.at(1);
            // normal case
            if (run != -1) {
                // Get "s"
                // debug
                cout << "            dct_coeff_next: ";
                dct_zz_i = dct_zz_i + run +1;
                int s = read_bits(1);
                if (s == 0) {
                    dct_zz.at(dct_zz_i) = level;
                    // debug
                    cout << "Run: " << run;
                    cout << " Level: ";
                    cout << dec << level << endl;
                } else if (s == 1) {
                    int neg_level = - level;
                    dct_zz.at(dct_zz_i) = neg_level;
                    // debug
                    cout << "Run: " << run;
                    cout << " Level: ";
                    cout << dec << neg_level << endl;
                }
                // debug
                cout << "                    --> get s : ";
                print_hex(s);            
                break; 
            } else {
                // escape case
                run = get_escape_run();
                level = get_escape_level();
                dct_zz_i = dct_zz_i + run +1;
                dct_zz.at(dct_zz_i) = level;
                // debug
                cout << "            dct_coeff_next: ";
                cout << "ESCAPE Run: " << run;
                cout << " Level: ";
                cout << dec << level << endl;
            }
        }
    }
}

void Decoder::dct_coeff_next_s() {
    uint64_t nbs64 = 0;
    uint32_t second_nbs32 = 0;
    for (int i=2; i<29; i++) {
        second_nbs32 = nextbits(i);
        nbs64 = i;
        nbs64 = (nbs64 << 32) + second_nbs32;
        // Lookup table via switch optimization
        uint8_t _ = 0;
        switch (nbs64) {
            case 0x1000000010:
                _ = read_bits(i);
                fill_dct_zz(1, 18);
                i = 29;
                break;
            case 0x1000000011:
                _ = read_bits(i);
                fill_dct_zz(1, 17);
                i = 29;
                break;
            case 0x1000000012:
                _ = read_bits(i);
                fill_dct_zz(1, 16);
                i = 29;
                break;
            case 0x1000000013:
                _ = read_bits(i);
                fill_dct_zz(1, 15);
                i = 29;
                break;
            case 0x1000000014:
                _ = read_bits(i);
                fill_dct_zz(6, 3);
                i = 29;
                break;
            case 0x1000000015:
                _ = read_bits(i);
                fill_dct_zz(16, 2);
                i = 29;
                break;
            case 0x1000000016:
                _ = read_bits(i);
                fill_dct_zz(15, 2);
                i = 29;
                break;
            case 0x1000000017:
                _ = read_bits(i);
                fill_dct_zz(14, 2);
                i = 29;
                break;
            case 0x1000000018:
                _ = read_bits(i);
                fill_dct_zz(13, 2);
                i = 29;
                break;
            case 0x1000000019:
                _ = read_bits(i);
                fill_dct_zz(12, 2);
                i = 29;
                break;
            case 0x100000001a:
                _ = read_bits(i);
                fill_dct_zz(11, 2);
                i = 29;
                break;
            case 0x100000001b:
                _ = read_bits(i);
                fill_dct_zz(31, 1);
                i = 29;
                break;
            case 0x100000001c:
                _ = read_bits(i);
                fill_dct_zz(30, 1);
                i = 29;
                break;
            case 0x100000001d:
                _ = read_bits(i);
                fill_dct_zz(29, 1);
                i = 29;
                break;
            case 0x100000001e:
                _ = read_bits(i);
                fill_dct_zz(28, 1);
                i = 29;
                break;
            case 0x100000001f:
                _ = read_bits(i);
                fill_dct_zz(27, 1);
                i = 29;
                break;
            case 0xf00000010:
                _ = read_bits(i);
                fill_dct_zz(0, 40);
                i = 29;
                break;
            case 0xf00000011:
                _ = read_bits(i);
                fill_dct_zz(0, 39);
                i = 29;
                break;
            case 0xf00000012:
                _ = read_bits(i);
                fill_dct_zz(0, 38);
                i = 29;
                break;
            case 0xf00000013:
                _ = read_bits(i);
                fill_dct_zz(0, 37);
                i = 29;
                break;
            case 0xf00000014:
                _ = read_bits(i);
                fill_dct_zz(0, 36);
                i = 29;
                break;
            case 0xf00000015:
                _ = read_bits(i);
                fill_dct_zz(0, 35);
                i = 29;
                break;
            case 0xf00000016:
                _ = read_bits(i);
                fill_dct_zz(0, 34);
                i = 29;
                break;
            case 0xf00000017:
                _ = read_bits(i);
                fill_dct_zz(0, 33);
                i = 29;
                break;
            case 0xf00000018:
                _ = read_bits(i);
                fill_dct_zz(0, 32);
                i = 29;
                break;
            case 0xf00000019:
                _ = read_bits(i);
                fill_dct_zz(1, 14);
                i = 29;
                break;
            case 0xf0000001a:
                _ = read_bits(i);
                fill_dct_zz(1, 13);
                i = 29;
                break;
            case 0xf0000001b:
                _ = read_bits(i);
                fill_dct_zz(1, 12);
                i = 29;
                break;
            case 0xf0000001c:
                _ = read_bits(i);
                fill_dct_zz(1, 11);
                i = 29;
                break;
            case 0xf0000001d:
                _ = read_bits(i);
                fill_dct_zz(1, 10);
                i = 29;
                break;
            case 0xf0000001e:
                _ = read_bits(i);
                fill_dct_zz(1, 9);
                i = 29;
                break;
            case 0xf0000001f:
                _ = read_bits(i);
                fill_dct_zz(1, 8);
                i = 29;
                break;
            case 0xe00000010:
                _ = read_bits(i);
                fill_dct_zz(0, 31);
                i = 29;
                break;
            case 0xe00000011:
                _ = read_bits(i);
                fill_dct_zz(0, 30);
                i = 29;
                break;
            case 0xe00000012:
                _ = read_bits(i);
                fill_dct_zz(0, 29);
                i = 29;
                break;
            case 0xe00000013:
                _ = read_bits(i);
                fill_dct_zz(0, 28);
                i = 29;
                break;
            case 0xe00000014:
                _ = read_bits(i);
                fill_dct_zz(0, 27);
                i = 29;
                break;
            case 0xe00000015:
                _ = read_bits(i);
                fill_dct_zz(0, 26);
                i = 29;
                break;
            case 0xe00000016:
                _ = read_bits(i);
                fill_dct_zz(0, 25);
                i = 29;
                break;
            case 0xe00000017:
                _ = read_bits(i);
                fill_dct_zz(0, 24);
                i = 29;
                break;
            case 0xe00000018:
                _ = read_bits(i);
                fill_dct_zz(0, 23);
                i = 29;
                break;
            case 0xe00000019:
                _ = read_bits(i);
                fill_dct_zz(0, 22);
                i = 29;
                break;
            case 0xe0000001a:
                _ = read_bits(i);
                fill_dct_zz(0, 21);
                i = 29;
                break;
            case 0xe0000001b:
                _ = read_bits(i);
                fill_dct_zz(0, 20);
                i = 29;
                break;
            case 0xe0000001c:
                _ = read_bits(i);
                fill_dct_zz(0, 19);
                i = 29;
                break;
            case 0xe0000001d:
                _ = read_bits(i);
                fill_dct_zz(0, 18);
                i = 29;
                break;
            case 0xe0000001e:
                _ = read_bits(i);
                fill_dct_zz(0, 17);
                i = 29;
                break;
            case 0xe0000001f:
                _ = read_bits(i);
                fill_dct_zz(0, 16);
                i = 29;
                break;
            case 0xd00000010:
                _ = read_bits(i);
                fill_dct_zz(10, 2);
                i = 29;
                break;
            case 0xd00000011:
                _ = read_bits(i);
                fill_dct_zz(9, 2);
                i = 29;
                break;
            case 0xd00000012:
                _ = read_bits(i);
                fill_dct_zz(5, 3);
                i = 29;
                break;
            case 0xd00000013:
                _ = read_bits(i);
                fill_dct_zz(3, 4);
                i = 29;
                break;
            case 0xd00000014:
                _ = read_bits(i);
                fill_dct_zz(2, 5);
                i = 29;
                break;
            case 0xd00000015:
                _ = read_bits(i);
                fill_dct_zz(1, 7);
                i = 29;
                break;
            case 0xd00000016:
                _ = read_bits(i);
                fill_dct_zz(1, 6);
                i = 29;
                break;
            case 0xd00000017:
                _ = read_bits(i);
                fill_dct_zz(0, 15);
                i = 29;
                break;
            case 0xd00000018:
                _ = read_bits(i);
                fill_dct_zz(0, 14);
                i = 29;
                break;
            case 0xd00000019:
                _ = read_bits(i);
                fill_dct_zz(0, 13);
                i = 29;
                break;
            case 0xd0000001a:
                _ = read_bits(i);
                fill_dct_zz(0, 12);
                i = 29;
                break;
            case 0xd0000001b:
                _ = read_bits(i);
                fill_dct_zz(26, 1);
                i = 29;
                break;
            case 0xd0000001c:
                _ = read_bits(i);
                fill_dct_zz(25, 1);
                i = 29;
                break;
            case 0xd0000001d:
                _ = read_bits(i);
                fill_dct_zz(24, 1);
                i = 29;
                break;
            case 0xd0000001e:
                _ = read_bits(i);
                fill_dct_zz(23, 1);
                i = 29;
                break;
            case 0xd0000001f:
                _ = read_bits(i);
                fill_dct_zz(22, 1);
                i = 29;
                break;
            case 0xc00000010:
                _ = read_bits(i);
                fill_dct_zz(0, 11);
                i = 29;
                break;
            case 0xc00000011:
                _ = read_bits(i);
                fill_dct_zz(8, 2);
                i = 29;
                break;
            case 0xc00000012:
                _ = read_bits(i);
                fill_dct_zz(4, 3);
                i = 29;
                break;
            case 0xc00000013:
                _ = read_bits(i);
                fill_dct_zz(0, 10);
                i = 29;
                break;
            case 0xc00000014:
                _ = read_bits(i);
                fill_dct_zz(2, 4);
                i = 29;
                break;
            case 0xc00000015:
                _ = read_bits(i);
                fill_dct_zz(7, 2);
                i = 29;
                break;
            case 0xc00000016:
                _ = read_bits(i);
                fill_dct_zz(21, 1);
                i = 29;
                break;
            case 0xc00000017:
                _ = read_bits(i);
                fill_dct_zz(20, 1);
                i = 29;
                break;
            case 0xc00000018:
                _ = read_bits(i);
                fill_dct_zz(0, 9);
                i = 29;
                break;
            case 0xc00000019:
                _ = read_bits(i);
                fill_dct_zz(19, 1);
                i = 29;
                break;
            case 0xc0000001a:
                _ = read_bits(i);
                fill_dct_zz(18, 1);
                i = 29;
                break;
            case 0xc0000001b:
                _ = read_bits(i);
                fill_dct_zz(1, 5);
                i = 29;
                break;
            case 0xc0000001c:
                _ = read_bits(i);
                fill_dct_zz(3, 3);
                i = 29;
                break;
            case 0xc0000001d:
                _ = read_bits(i);
                fill_dct_zz(0, 8);
                i = 29;
                break;
            case 0xc0000001e:
                _ = read_bits(i);
                fill_dct_zz(6, 2);
                i = 29;
                break;
            case 0xc0000001f:
                _ = read_bits(i);
                fill_dct_zz(17, 1);
                i = 29;
                break;
            case 0xa00000008:
                _ = read_bits(i);
                fill_dct_zz(16, 1);
                i = 29;
                break;
            case 0xa00000009:
                _ = read_bits(i);
                fill_dct_zz(5, 2);
                i = 29;
                break;
            case 0xa0000000a:
                _ = read_bits(i);
                fill_dct_zz(0, 7);
                i = 29;
                break;
            case 0xa0000000b:
                _ = read_bits(i);
                fill_dct_zz(2, 3);
                i = 29;
                break;
            case 0xa0000000c:
                _ = read_bits(i);
                fill_dct_zz(1, 4);
                i = 29;
                break;
            case 0xa0000000d:
                _ = read_bits(i);
                fill_dct_zz(15, 1);
                i = 29;
                break;
            case 0xa0000000e:
                _ = read_bits(i);
                fill_dct_zz(14, 1);
                i = 29;
                break;
            case 0xa0000000f:
                _ = read_bits(i);
                fill_dct_zz(4, 2);
                i = 29;
                break;
            case 0x600000001:
                _ = read_bits(i);
                fill_dct_zz(-1, -1);
                i = 29;
                break;
            case 0x700000004:
                _ = read_bits(i);
                fill_dct_zz(2, 2);
                i = 29;
                break;
            case 0x700000005:
                _ = read_bits(i);
                fill_dct_zz(9, 1);
                i = 29;
                break;
            case 0x700000006:
                _ = read_bits(i);
                fill_dct_zz(0, 4);
                i = 29;
                break;
            case 0x700000007:
                _ = read_bits(i);
                fill_dct_zz(8, 1);
                i = 29;
                break;
            case 0x600000004:
                _ = read_bits(i);
                fill_dct_zz(7, 1);
                i = 29;
                break;
            case 0x600000005:
                _ = read_bits(i);
                fill_dct_zz(6, 1);
                i = 29;
                break;
            case 0x600000006:
                _ = read_bits(i);
                fill_dct_zz(1, 2);
                i = 29;
                break;
            case 0x600000007:
                _ = read_bits(i);
                fill_dct_zz(5, 1);
                i = 29;
                break;
            case 0x800000020:
                _ = read_bits(i);
                fill_dct_zz(13, 1);
                i = 29;
                break;
            case 0x800000021:
                _ = read_bits(i);
                fill_dct_zz(0, 6);
                i = 29;
                break;
            case 0x800000022:
                _ = read_bits(i);
                fill_dct_zz(12, 1);
                i = 29;
                break;
            case 0x800000023:
                _ = read_bits(i);
                fill_dct_zz(11, 1);
                i = 29;
                break;
            case 0x800000024:
                _ = read_bits(i);
                fill_dct_zz(3, 2);
                i = 29;
                break;
            case 0x800000025:
                _ = read_bits(i);
                fill_dct_zz(1, 3);
                i = 29;
                break;
            case 0x800000026:
                _ = read_bits(i);
                fill_dct_zz(0, 5);
                i = 29;
                break;
            case 0x800000027:
                _ = read_bits(i);
                fill_dct_zz(10, 1);
                i = 29;
                break;
            case 0x500000005:
                _ = read_bits(i);
                fill_dct_zz(0, 3);
                i = 29;
                break;
            case 0x500000006:
                _ = read_bits(i);
                fill_dct_zz(4, 1);
                i = 29;
                break;
            case 0x500000007:
                _ = read_bits(i);
                fill_dct_zz(3, 1);
                i = 29;
                break;
            case 0x400000004:
                _ = read_bits(i);
                fill_dct_zz(0, 2);
                i = 29;
                break;
            case 0x400000005:
                _ = read_bits(i);
                fill_dct_zz(2, 1);
                i = 29;
                break;
            case 0x300000003:
                _ = read_bits(i);
                fill_dct_zz(1, 1);
                i = 29;
                break;
            case 0x200000003:
                _ = read_bits(i);
                fill_dct_zz(0, 1);
                i = 29;
                break;
        }
    }
}

void Decoder::fill_dct_zz(int run, int level) {
//    cout << "--- Match !!! ---" << endl;
    // normal case
    if (run != -1) {
        dct_zz_i = dct_zz_i + run +1;
        int s = read_bits(1);
        if (s == 0) {
            dct_zz.at(dct_zz_i) = level;
            // debug
//            cout << "level : " << level << endl;
        } else if (s == 1) {
            int neg_level = - level;
            dct_zz.at(dct_zz_i) = neg_level;
            // debug
//            cout << "level : " << neg_level << endl;            
        }
    } else {
        // escape case
        run = get_escape_run();
        level = get_escape_level();
        dct_zz_i = dct_zz_i + run +1;
        dct_zz.at(dct_zz_i) = level;
        // debug
//        cout << "level : " << level << endl;
    }  
}

int Decoder::get_escape_run() {
    int vlc_code = read_bits(6);
    return vlc_code;
}

int Decoder::get_escape_level() {
    uint8_t first8_code = read_bits(8);
    if ((first8_code == 0x80) || (first8_code == 0)) {
        // 16 bits
        uint8_t last8_code = read_bits(8);
        uint16_t total_code = first8_code;
        total_code = (total_code << 8) + last8_code;
        if (first8_code == 0) { 
            // pos
            int pos_total_code = total_code;
            return pos_total_code;
        } else {
            // neg
//            uint16_t tmp = ~(total_code - 1);
            uint8_t tmp_lc = ~(last8_code - 1);
            uint16_t tmp = 0;
            tmp += tmp_lc;
            int neg_total_code = - tmp;
//            cout << "neg val : " << neg_total_code << endl;
            return neg_total_code;
        }
    } else {
        // 8 bits
        if (first8_code > 0x80) {
            // neg
            uint8_t tmp = ~(first8_code - 1);
            int neg_first8_code = - tmp;
            return neg_first8_code;
        } else {
            // pos
            int pos_first8_code = first8_code;
            return pos_first8_code;
        }
    }
    
    
    
    

}

void Decoder::reconstruct_dct(int num) {
//    cout << "-------- dct_zz --------" << endl;
//    for (int i=0; i<64; i++) {
//        cout << dct_zz.at(i) << " ";
//        if ((i+1) % 8 == 0) {
//            cout << endl;
//        }
//    }
//    cout << endl;

    if (mb_intra == "1") {
        int i = 0;
        for (int m=0; m<8; m++) {
            for (int n=0; n<8; n++) {
                i = zigzag_m[m][n];
                // dct_recon[m][n] = ( 2 * dct_zz[i] * quantizer_scale * intra_quant[m][n] ) /16;
                dct_recon[m][n] = (dct_zz[i] * quantizer_scale * intra_quant[m][n]) >> 3;
                 if ( ( dct_recon[m][n] & 1 ) == 0 ) {
                     dct_recon[m][n] = dct_recon[m][n] - sign(dct_recon[m][n]);
                 }
                if (dct_recon[m][n] > 2047) {
                    dct_recon[m][n] = 2047;
                }
                if (dct_recon[m][n] < -2048) {
                    dct_recon[m][n] = -2048;
                }
            }
        }
//        if ((num <= 3) && (num >= 0)) {
//            // dct_recon[0][0] = dct_dc_y_past + dct_zz[0] * 8;
//            dct_recon[0][0] = dct_dc_y_past + (dct_zz[0] << 3);
//            dct_dc_y_past = dct_recon[0][0];
//        } else if (num == 4) {
//            // dct_recon[0][0] = dct_zz[0] * 8;
//            dct_recon[0][0] = dct_zz[0] << 3;
//            if (( mb_address - past_intra_address > 1)) {
//                // dct_recon[0][0] = 128 * 8 + dct_recon[0][0];
//                dct_recon[0][0] = 1024 + dct_recon[0][0];
//            } else {
//                dct_recon[0][0] = dct_dc_cb_past + dct_recon[0][0] ;
//            }
//            dct_dc_cb_past = dct_recon[0][0];
//        } else {
//            // dct_recon[0][0] = dct_zz[0] * 8 ;
//            dct_recon[0][0] = dct_zz[0] << 3 ;
//            if ((mb_address - past_intra_address > 1)) {
//                dct_recon[0][0] = 1024 + dct_recon[0][0];
//            } else {
//                dct_recon[0][0] = dct_dc_cr_past + dct_recon[0][0];
//            }
//            dct_dc_cr_past = dct_recon[0][0];
//        }
        switch (num) {
            case 0:
            case 1:
            case 2:
            case 3:
                dct_recon[0][0] = dct_dc_y_past + (dct_zz[0] << 3);
                dct_dc_y_past = dct_recon[0][0];
                break;
            case 4:
                dct_recon[0][0] = dct_zz[0] << 3;
                if (( mb_address - past_intra_address > 1)) {
                    dct_recon[0][0] = 1024 + dct_recon[0][0];
                } else {
                    dct_recon[0][0] = dct_dc_cb_past + dct_recon[0][0] ;
                }
                dct_dc_cb_past = dct_recon[0][0];
                break;
            case 5:
                dct_recon[0][0] = dct_zz[0] << 3 ;
                if ((mb_address - past_intra_address > 1)) {
                    dct_recon[0][0] = 1024 + dct_recon[0][0];
                } else {
                    dct_recon[0][0] = dct_dc_cr_past + dct_recon[0][0];
                }
                dct_dc_cr_past = dct_recon[0][0];
                break;
        }
    } else {
        int i = 0;
        for (int m=0; m<8; m++) {
            for (int n=0; n<8; n++) {
                i = zigzag_m[m][n];
                // dct_recon[m][n] = ( ( (2 * dct_zz[i]) + sign(dct_zz[i]) ) * quantizer_scale * non_intra_quant[m][n] ) / 16;
                // dct_recon[m][n] = ((dct_zz[i] + sign(dct_zz[i])) * quantizer_scale * non_intra_quant[m][n] ) * 0.125;
                dct_recon[m][n] = ((dct_zz[i] + sign(dct_zz[i])) * quantizer_scale * non_intra_quant[m][n] ) >> 3;
                if ( ( dct_recon[m][n] & 1 ) == 0 ) {
                    dct_recon[m][n] = dct_recon[m][n] - sign(dct_recon[m][n]);
                }
                if (dct_recon[m][n] > 2047) {
                    dct_recon[m][n] = 2047;
                }
                if (dct_recon[m][n] < -2048) {
                    dct_recon[m][n] = -2048;
                }
                if ( dct_zz[i] == 0 ) {
                    dct_recon[m][n] = 0;
                }
            }
        }
    }

//    cout << "-------- dct_recon --------" << endl;
//    for (int i=0; i<8; i++) {
//        for (int j=0; j<8; j++) {
//            cout << dct_recon[i][j] << " ";
//            if (j == 7) {
//                cout << endl;
//            }
//        }
//    }
//    cout << endl;
}

void Decoder::idct() {
    vector<vector<int>> idct_prosessor (8, vector<int>(8, 0));
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            double idct_val = 0;
             for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    idct_val += idct_table.at(x).at(i) * idct_table.at(y).at(j) * dct_recon.at(i).at(j);
                }
            }
//            if (picture_coding_type == 1) {
//                if ((int)idct_val < 0) {
//                    idct_prosessor.at(x).at(y) = 0;
//                } else if ((int)idct_val > 255) {
//                    idct_prosessor.at(x).at(y) = 255;
//                } else {
//                    idct_prosessor.at(x).at(y) = (int)idct_val;
//                }     
//            } else if ((picture_coding_type == 2) || (picture_coding_type == 3)) {
//                idct_prosessor.at(x).at(y) = (int)idct_val;
//            }
            switch (picture_coding_type) {
                case 1:
                    if ((int)idct_val < 0) {
                        idct_prosessor.at(x).at(y) = 0;
                    } else if ((int)idct_val > 255) {
                        idct_prosessor.at(x).at(y) = 255;
                    } else {
                        idct_prosessor.at(x).at(y) = (int)idct_val;
                    }
                    break;
                case 2:
                case 3:
                    idct_prosessor.at(x).at(y) = (int)idct_val;
                    break;
            }
        }
    }
    dct_recon = idct_prosessor;
    // debug
//    cout << "-------- idct --------" << endl;
//    for (int i = 0; i < 8; i++) {
//        cout << endl;
//        for (int j = 0; j < 8; j++) {
//            cout << dct_recon.at(i).at(j) << " ";
//        }
//    }
//    cout << endl;
    // Pushback to pic_mb_vec
//    if (picture_coding_type == 1) {
//        pic_mb_vec.push_back(dct_recon);
//    } else if ((picture_coding_type == 2) || (picture_coding_type == 3)) {
//        mb_p_num.push_back(mb_address);
//        mp_p_i.push_back(block_i);
//        pic_mb_vec_p.push_back(dct_recon);
//    }
    switch (picture_coding_type) {
        case 1:
            pic_mb_vec.push_back(dct_recon);
            break;
        case 2:
        case 3:
            mb_p_num.push_back(mb_address);
            mp_p_i.push_back(block_i);
            pic_mb_vec_p.push_back(dct_recon);        
            break;
    }
}

void Decoder::recon_pic() {
    vector<vector<int>>y_result (v_size, vector<int>(h_size, 0));
    vector<vector<int>>cb_result (v_size / 2, vector<int>(h_size / 2, 0));
    vector<vector<int>>cr_result (v_size / 2, vector<int>(h_size / 2, 0));
    if (picture_coding_type == 1) {
        int y01_r = 0;
        int y01_c = 0;
        int y23_r = 8;
        int y23_c = 0;
        int cb_r = 0;
        int cb_c = 0;
        int cr_r = 0;
        int cr_c = 0;
        // counter
        int y01_acc = 0;
        int y23_acc = 0;
        int cb_acc = 0;
        int cr_acc = 0;
        int mb_num = pic_mb_vec.size();
        for (int i=0; i<mb_num; i++) {
            if ((((i + 1) % 6) >= 1) && (((i + 1) % 6) <= 2)) {
                // Y0, Y1
                for (int r=0; r<8; r++) {
                    for (int c=0; c<8; c++) {
                        y_result.at(y01_r + r).at(y01_c + c) = pic_mb_vec.at(i).at(r).at(c);
                    }
                }
                // update row & col
                if ((y01_acc + 1) % (mb_width * 2) == 0) {
                    y01_r += 16;
                    y01_c = 0;
                } else {
                    y01_c += 8;
                }
                // update counter
                y01_acc += 1;
            } else if ((((i + 1) % 6) >= 3) && (((i + 1) % 6) <= 4)) {
                // Y2, Y3
                for (int r=0; r<8; r++) {
                    for (int c=0; c<8; c++) {
                        y_result.at(y23_r + r).at(y23_c + c) = pic_mb_vec.at(i).at(r).at(c);
                    }
                }
                // update row & col
                if ((y23_acc + 1) % (mb_width * 2) == 0) {
                    y23_r += 16;
                    y23_c = 0;
                } else {
                    y23_c += 8;
                }
                // update counter
                y23_acc += 1;
            } else if (((i + 1) % 6) == 5) {
                // Cb
                for (int r=0; r<8; r++) {
                    for (int c=0; c<8; c++) {
                        cb_result.at(cb_r + r).at(cb_c + c) = pic_mb_vec.at(i).at(r).at(c);
                    }
                }
                // update row & col
                if ((cb_acc + 1) % mb_width == 0) {
                    cb_r += 8;
                    cb_c = 0;
                } else {
                    cb_c += 8;
                }
               // update counter
                cb_acc += 1;
            } else if (((i + 1) % 6) == 0) {
                // Cr
                for (int r=0; r<8; r++) {
                    for (int c=0; c<8; c++) {
                        cr_result.at(cr_r + r).at(cr_c + c) = pic_mb_vec.at(i).at(r).at(c);
                    }
                }
                // update row & col
                if ((cr_acc + 1) % mb_width == 0) {
                    cr_r += 8;
                    cr_c = 0;
                } else {
                    cr_c += 8;
                }
               // update counter
                cr_acc += 1;
            }
        }
        y_result_final = y_result;
        cb_result_final = cb_result;
        cr_result_final = cr_result;
        // reset
        pic_mb_vec = {};
    } else {
        for (int i=0; i<pic_mb_vec_p.size(); i++) {
            int mb_num = mb_p_num.at(i);
            int b_i = mp_p_i.at(i);
            // int y_r = (mb_num / mb_width) * 16;
            int y_r = (mb_num / mb_width) << 4;
            // int y_c = (mb_num % mb_width) * 16;
            int y_c = (mb_num % mb_width) << 4;
            // int c_r = (mb_num / mb_width) * 8;
            int c_r = (mb_num / mb_width) << 3;
            // int c_c = (mb_num % mb_width) * 8;
            int c_c = (mb_num % mb_width) << 3;
            switch (b_i) {
                case 0:
                    for (int r=0; r<8; r++) {
                        for (int c=0; c<8; c++) {
                            y_result.at(y_r + r).at(y_c + c) = pic_mb_vec_p.at(i).at(r).at(c);
                        }
                    }
                    break;
                case 1:
                    for (int r=0; r<8; r++) {
                        for (int c=0; c<8; c++) {
                            y_result.at(y_r + r).at(y_c + 8 + c) = pic_mb_vec_p.at(i).at(r).at(c);                         
                        }
                    }                
                    break;
                case 2:
                    for (int r=0; r<8; r++) {
                        for (int c=0; c<8; c++) {
                            y_result.at(y_r + 8 + r).at(y_c + c) = pic_mb_vec_p.at(i).at(r).at(c);                  
                        }
                    }
                    break;
                case 3:
                    for (int r=0; r<8; r++) {
                        for (int c=0; c<8; c++) {
                            y_result.at(y_r + 8 + r).at(y_c + 8 + c) = pic_mb_vec_p.at(i).at(r).at(c);               
                        }
                    }
                    break;
                case 4:              
                    for (int r=0; r<8; r++) {
                        for (int c=0; c<8; c++) {
                            cb_result.at(c_r + r).at(c_c + c) = pic_mb_vec_p.at(i).at(r).at(c);                              
                        }
                    }
                    break;
                case 5:               
                    for (int r=0; r<8; r++) {
                        for (int c=0; c<8; c++) {
                            cr_result.at(c_r + r).at(c_c + c) = pic_mb_vec_p.at(i).at(r).at(c);                             
                        }
                    }
                    break;
            }
        }
        // update result to result all
        y_result_final = y_result;
        cb_result_final = cb_result;
        cr_result_final = cr_result; 
        // reset 
        mb_p_num = {};
        mp_p_i = {};
        pic_mb_vec_p = {};
    }
}

void Decoder::ycbcr2rgb(bool to_buffer, bool to_output) {
    int img_y_w = y_result_final.at(0).size();
    int img_y_h = y_result_final.size();
    double R;
    double G;
    double B;
    for (int r = 0; r < v_size; r++) {
        for (int c = 0; c < h_size; c++) {
            // For Cb, Cr
            int c_r = r / 2;
            int c_c = c / 2;
            // Get data
            double Y = y_result_final.at(r).at(c);
            double Cb = cb_result_final.at(c_r).at(c_c);
            double Cr = cr_result_final.at(c_r).at(c_c);

            if (mb_intra_vec.at(r).at(c) == "1") {
                R = Y + (1.402 * (Cr - 128));
                G = Y - (0.344 * (Cb - 128)) - (0.714 * (Cr - 128));
                B = Y + (1.772 * (Cb - 128));
            } else {
                R = pel_R.at(r).at(c);
                G = pel_G.at(r).at(c);
                B = pel_B.at(r).at(c);
                // R += ((255/219.0) * Y) + ((255/224.0) * 1.402 * Cr);
                // G += ((255/219.0) * Y) - ((255/224.0) * 1.772 * (0.114/0.587) * Cb) - ((255/224.0) * 1.402 * (0.299/0.587) * Cr);
                // B += ((255/219.0) * Y) + ((255/224.0) * 1.772 * Cb);
                R += (1.164 * Y) + (1.596 * Cr);
                G += (1.164 * Y) - (0.392 * Cb) - (0.813 * Cr);
                B += (1.164 * Y) + (2.017 * Cb);
            }
            // Clip range to 0 - 255
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
            // update pel_past
            pel_tmp_R.at(r).at(c) = (int)R;
            pel_tmp_G.at(r).at(c) = (int)G;
            pel_tmp_B.at(r).at(c) = (int)B;
        }
    }
    if (to_buffer) {
        pel_past_R = pel_tmp_R;
        pel_past_G = pel_tmp_G;
        pel_past_B = pel_tmp_B;
    }
    if (to_output) {
//        img_queue.push_back(pel_tmp_R);
//        img_queue.push_back(pel_tmp_G);
//        img_queue.push_back(pel_tmp_B);
        // push to buffer
        rgb2cvmat(pel_tmp_R, pel_tmp_G, pel_tmp_B);            
    }
}

void Decoder::output_img() {
    // Declare output img class
    BMP Bmp_image;
    Bmp_image.SetSize(320, 240);
    for (int r = 0; r < v_size; r++) {
        for (int c = 0; c < h_size; c++) {
            Bmp_image (c, r) -> Red = img_queue.at(0).at(r).at(c);
            Bmp_image (c, r) -> Green = img_queue.at(1).at(r).at(c);
            Bmp_image (c, r) -> Blue = img_queue.at(2).at(r).at(c);
            Bmp_image (c, r) -> Alpha = 0;
        }
    }
    string wfilename = "C:\\Users\\o1r2g\\OneDrive\\Desktop\\cpp_tutorial\\CPPWorkspace\\MPEG1_decoder\\pictures\\MPEG_" + to_string(pic_num) + ".bmp";
    Bmp_image.WriteToFile(wfilename.c_str());
    img_queue.erase(img_queue.begin(), img_queue.begin()+3);
}

void Decoder::output_bmp() {
    int img_y_w = y_result_final.at(0).size();
    int img_y_h = y_result_final.size();
    double R;
    double G;
    double B;
    // Declare output img class
    BMP Bmp_image;
    Bmp_image.SetSize(320, 240);
    // Bmp_image.SetBitDepth(24);
    for (int r = 0; r < v_size; r++) {
        for (int c = 0; c < h_size; c++) {
            // For Cb, Cr
            int c_r = r / 2;
            int c_c = c / 2;
            // Get data
            double Y = y_result_final.at(r).at(c);
            double Cb = cb_result_final.at(c_r).at(c_c);
            double Cr = cr_result_final.at(c_r).at(c_c);

            if (mb_intra_vec.at(r).at(c) == "1") {
                R = Y + (1.402 * (Cr - 128));
                G = Y - (0.344 * (Cb - 128)) - (0.714 * (Cr - 128));
                B = Y + (1.772 * (Cb - 128));
                
//                if ((r == (int)(295 / 16)) && (c == (int)(295 % 16))) {
//                    cout << "(Y, Cb, Cr) : " << Y << ", " << Cb << ", " << Cr << endl;
//                    cout << "(R, G, B) : " << R << ", " << G << ", " << B << endl;
//                }
                
            } else {
                R = pel_R.at(r).at(c);
                G = pel_G.at(r).at(c);
                B = pel_B.at(r).at(c);
//                R += ((255/219.0) * Y) + ((255/224.0) * 1.402 * Cr);
//                G += ((255/219.0) * Y) - ((255/224.0) * 1.772 * (0.114/0.587) * Cb) - ((255/224.0) * 1.402 * (0.299/0.587) * Cr);
//                B += ((255/219.0) * Y) + ((255/224.0) * 1.772 * Cb);
                R += (1.164 * Y) + (1.596 * Cr);
                G += (1.164 * Y) - (0.392 * Cb) - (0.813 * Cr);
                B += (1.164 * Y) + (2.017 * Cb);                
            }
            
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
            // update pel_past
            pel_past_R.at(r).at(c) = (int)R;
            pel_past_G.at(r).at(c) = (int)G;
            pel_past_B.at(r).at(c) = (int)B;
        }
    }
    string wfilename = "C:\\Users\\o1r2g\\OneDrive\\Desktop\\cpp_tutorial\\CPPWorkspace\\MPEG1_decoder\\pictures\\MPEG_" + to_string(pic_num) + ".bmp";
    Bmp_image.WriteToFile(wfilename.c_str());
}

void Decoder::rgb2cvmat(vector<vector<int>> cv_R, vector<vector<int>> cv_G, vector<vector<int>> cv_B) {
    Mat image(240, 320, CV_8UC3);
    for (int r = 0; r< 240; r++)
    {
        for (int c = 0; c< 320; c++)
        {
            image.at<Vec3b>(r, c)[0] = cv_B.at(r).at(c);
            image.at<Vec3b>(r, c)[1] = cv_G.at(r).at(c);
            image.at<Vec3b>(r, c)[2] = cv_R.at(r).at(c);
        }
    }
    imageQueue.push_back(image);
}

// Reconstruct P-frame
void Decoder::cal_motion_vector_p() {
    if (forward_f == 1 || motion_horizontal_forward_code == 0) {
        complement_horizontal_forward_r = 0;
    } else {
        complement_horizontal_forward_r = forward_f - 1 - motion_horizontal_forward_r;
    }
    if (forward_f == 1 || motion_vertical_forward_code == 0) {
        complement_vertical_forward_r = 0;
    } else {
        complement_vertical_forward_r = forward_f - 1 - motion_vertical_forward_r;
    }
    int right_big = 0;
    int right_little = motion_horizontal_forward_code * forward_f;
    if (right_little == 0) {
        right_big = 0;
    } else {
        if (right_little > 0) {
            right_little = right_little - complement_horizontal_forward_r;
            right_big = right_little - 32 * forward_f;
        } else {
            right_little = right_little + complement_horizontal_forward_r;
            right_big = right_little + 32 * forward_f;
        }
    }
    int down_big = 0;
    int down_little = motion_vertical_forward_code * forward_f;
    if (down_little == 0) {
        down_big = 0;
    } else {
        if (down_little > 0) {
            down_little = down_little - complement_vertical_forward_r;
            down_big = down_little - 32 * forward_f;
        } else {
            down_little = down_little + complement_vertical_forward_r;
            down_big = down_little + 32 * forward_f;
        }
    }
    int max = ( 16 * forward_f ) - 1;
    int min = ( -16 * forward_f );
    int new_vector = recon_right_for_prev + right_little;
    if (new_vector <= max && new_vector >= min) {
        recon_right_for = recon_right_for_prev + right_little;
    } else {
        recon_right_for = recon_right_for_prev + right_big;
    }
    recon_right_for_prev = recon_right_for;
    if ( full_pel_forward_vector ) {
        recon_right_for = recon_right_for << 1;
    }
    new_vector = recon_down_for_prev + down_little;
    if ( new_vector <= max && new_vector >= min ) {
        recon_down_for = recon_down_for_prev + down_little;
    } else {
        recon_down_for = recon_down_for_prev + down_big;
    }
    recon_down_for_prev = recon_down_for;
    if ( full_pel_forward_vector ) {
        recon_down_for = recon_down_for << 1;
    }
}

void Decoder::cal_motion_vector_b() {
    if (backward_f == 1 || motion_horizontal_backward_code == 0) {
        complement_horizontal_backward_r = 0;
    } else {
        complement_horizontal_backward_r = backward_f - 1 - motion_horizontal_backward_r;
    }
    if (backward_f == 1 || motion_vertical_backward_code == 0) {
        complement_vertical_backward_r = 0;
    } else {
        complement_vertical_backward_r = backward_f - 1 - motion_vertical_backward_r;
    }
    int right_big = 0;
    int right_little = motion_horizontal_backward_code * backward_f;
    if (right_little == 0) {
        right_big = 0;
    } else {
        if (right_little > 0) {
            right_little = right_little - complement_horizontal_backward_r;
            right_big = right_little - 32 * backward_f;
        } else {
            right_little = right_little + complement_horizontal_backward_r;
            right_big = right_little + 32 * backward_f;
        }
    }
    int down_big = 0;
    int down_little = motion_vertical_backward_code * backward_f;
    if (down_little == 0) {
        down_big = 0;
    } else {
        if (down_little > 0) {
            down_little = down_little - complement_vertical_backward_r;
            down_big = down_little - 32 * backward_f;
        } else {
            down_little = down_little + complement_vertical_backward_r;
            down_big = down_little + 32 * backward_f;
        }
    }
    int max = ( 16 * backward_f ) - 1;
    int min = ( -16 * backward_f );     
    int new_vector = recon_right_bac_prev + right_little;    
    if (new_vector <= max && new_vector >= min) {
        recon_right_bac = recon_right_bac_prev + right_little;
    } else {
        recon_right_bac = recon_right_bac_prev + right_big;
    }
    recon_right_bac_prev = recon_right_bac;
    if ( full_pel_backward_vector ) {
        recon_right_bac = recon_right_bac << 1;
    }
    new_vector = recon_down_bac_prev + down_little;
    if ( new_vector <= max && new_vector >= min ) {
        recon_down_bac = recon_down_bac_prev + down_little;
    } else {
        recon_down_bac = recon_down_bac_prev + down_big;
    }
    recon_down_bac_prev = recon_down_bac;
    if ( full_pel_backward_vector ) {
        recon_down_bac = recon_down_bac << 1;
    }
}

void Decoder::decode_mv() {
    // final forward motion vector
    right_for = recon_right_for >> 1;
    down_for = recon_down_for >> 1;
    right_half_for = recon_right_for - 2 * right_for;
    down_half_for = recon_down_for - 2 * down_for;
    // final backward motion vector
    right_bac = recon_right_bac >> 1;
    down_bac = recon_down_bac >> 1;
    right_half_bac = recon_right_bac - 2 * right_bac;
    down_half_bac = recon_down_bac - 2 * down_bac;
    int R_for = 0;
    int G_for = 0;
    int B_for = 0;
    int R_bac = 0;
    int G_bac = 0;
    int B_bac = 0;    
    // RGB
    for (int i=0; i<16; i++) {
        for (int j=0; j<16; j++) {
            int pel_r = (mb_row * 16) + i;
            int pel_c = (mb_col * 16) + j;
            mb_intra_vec.at(pel_r).at(pel_c) = "0";

            if (mb_motion_forward == "1") {
                int pel_future_r = pel_r + down_for;
                int pel_future_c = pel_c + right_for;
//                cout << "(i, j) : " << i << ", " << j << endl;
//                cout << "(pel_r , pel_c) : " << pel_r << ", " << pel_c << endl;
//                cout << "(down_for, right_for) : " << down_for << ", " << right_for << endl;
//                cout << "(pel_past_r , pel_past_c) : " << pel_future_r << ", " << pel_future_c << endl;                
                if ( ! right_half_for && ! down_half_for ) {
                    // pel[i][j] = pel_past[i+down_for][j+right_for];
                    R_for = pel_future_R.at(pel_future_r).at(pel_future_c);
                    G_for = pel_future_G.at(pel_future_r).at(pel_future_c);
                    B_for = pel_future_B.at(pel_future_r).at(pel_future_c);
//                    pel_R.at(pel_r).at(pel_c) = pel_future_R.at(pel_future_r).at(pel_future_c);
//                    pel_G.at(pel_r).at(pel_c) = pel_future_G.at(pel_future_r).at(pel_future_c);
//                    pel_B.at(pel_r).at(pel_c) = pel_future_B.at(pel_future_r).at(pel_future_c);
                } else if ( ! right_half_for && down_half_for ) {
                    // pel[i][j] = ( pel_past[i+down_for][j+right_for] + pel_past[i+down_for+1][j+right_for] ) // 2;
                    R_for = round((pel_future_R.at(pel_future_r).at(pel_future_c) + pel_future_R.at(pel_future_r + 1).at(pel_future_c)) / 2);
                    G_for = round((pel_future_G.at(pel_future_r).at(pel_future_c) + pel_future_G.at(pel_future_r + 1).at(pel_future_c)) / 2);
                    B_for = round((pel_future_B.at(pel_future_r).at(pel_future_c) + pel_future_B.at(pel_future_r + 1).at(pel_future_c)) / 2);
//                    pel_R.at(pel_r).at(pel_c) = round((pel_future_R.at(pel_future_r).at(pel_future_c) + pel_future_R.at(pel_future_r + 1).at(pel_future_c)) / 2);
//                    pel_G.at(pel_r).at(pel_c) = round((pel_future_G.at(pel_future_r).at(pel_future_c) + pel_future_G.at(pel_future_r + 1).at(pel_future_c)) / 2);
//                    pel_B.at(pel_r).at(pel_c) = round((pel_future_B.at(pel_future_r).at(pel_future_c) + pel_future_B.at(pel_future_r + 1).at(pel_future_c)) / 2);
                } else if ( right_half_for && ! down_half_for ) {
                    // pel[i][j] = ( pel_past[i+down_for][j+right_for] + pel_past[i+down_for][j+right_for+1] ) // 2;
                    R_for = round((pel_future_R.at(pel_future_r).at(pel_future_c) + pel_future_R.at(pel_future_r).at(pel_future_c + 1)) / 2);
                    G_for = round((pel_future_G.at(pel_future_r).at(pel_future_c) + pel_future_G.at(pel_future_r).at(pel_future_c + 1)) / 2);
                    B_for = round((pel_future_B.at(pel_future_r).at(pel_future_c) + pel_future_B.at(pel_future_r).at(pel_future_c + 1)) / 2);
//                    pel_R.at(pel_r).at(pel_c) = round((pel_future_R.at(pel_future_r).at(pel_future_c) + pel_future_R.at(pel_future_r).at(pel_future_c + 1)) / 2);
//                    pel_G.at(pel_r).at(pel_c) = round((pel_future_G.at(pel_future_r).at(pel_future_c) + pel_future_G.at(pel_future_r).at(pel_future_c + 1)) / 2);
//                    pel_B.at(pel_r).at(pel_c) = round((pel_future_B.at(pel_future_r).at(pel_future_c) + pel_future_B.at(pel_future_r).at(pel_future_c + 1)) / 2);
                } else if ( right_half_for && down_half_for ) {
                    // pel[i][j] = ( pel_past[i+down_for][j+right_for] + pel_past[i+down_for+1][j+right_for] + pel_past[i+down_for][j+right_for+1] + pel_past[i+down_for+1][j+right_for+1] ) // 4;                
                    R_for = round((pel_future_R.at(pel_future_r).at(pel_future_c) + pel_future_R.at(pel_future_r + 1).at(pel_future_c) + pel_future_R.at(pel_future_r).at(pel_future_c + 1) + pel_future_R.at(pel_future_r + 1).at(pel_future_c + 1)) / 4);
                    G_for = round((pel_future_G.at(pel_future_r).at(pel_future_c) + pel_future_G.at(pel_future_r + 1).at(pel_future_c) + pel_future_G.at(pel_future_r).at(pel_future_c + 1) + pel_future_G.at(pel_future_r + 1).at(pel_future_c + 1)) / 4);
                    B_for = round((pel_future_B.at(pel_future_r).at(pel_future_c) + pel_future_B.at(pel_future_r + 1).at(pel_future_c) + pel_future_B.at(pel_future_r).at(pel_future_c + 1) + pel_future_B.at(pel_future_r + 1).at(pel_future_c + 1)) / 4);
//                    pel_R.at(pel_r).at(pel_c) = round((pel_future_R.at(pel_future_r).at(pel_future_c) + pel_future_R.at(pel_future_r + 1).at(pel_future_c) + pel_future_R.at(pel_future_r).at(pel_future_c + 1) + pel_future_R.at(pel_future_r + 1).at(pel_future_c + 1)) / 4);
//                    pel_G.at(pel_r).at(pel_c) = round((pel_future_G.at(pel_future_r).at(pel_future_c) + pel_future_G.at(pel_future_r + 1).at(pel_future_c) + pel_future_G.at(pel_future_r).at(pel_future_c + 1) + pel_future_G.at(pel_future_r + 1).at(pel_future_c + 1)) / 4);
//                    pel_B.at(pel_r).at(pel_c) = round((pel_future_B.at(pel_future_r).at(pel_future_c) + pel_future_B.at(pel_future_r + 1).at(pel_future_c) + pel_future_B.at(pel_future_r).at(pel_future_c + 1) + pel_future_B.at(pel_future_r + 1).at(pel_future_c + 1)) / 4);
                }
            }
            if (mb_motion_backward == "1") {
                int pel_past_r = pel_r + down_bac;
                int pel_past_c = pel_c + right_bac; 
//                cout << "(i, j) : " << i << ", " << j << endl;
//                cout << "(pel_r , pel_c) : " << pel_r << ", " << pel_c << endl;
//                cout << "(down_bac, right_bac) : " << down_bac << ", " << right_bac << endl;
//                cout << "(pel_past_r , pel_past_c) : " << pel_past_r << ", " << pel_past_c << endl;                  
                if ( ! right_half_bac && ! down_half_bac ) {             
                    // pel[i][j] = pel_past[i+down_for][j+right_for];
                    R_bac = pel_past_R.at(pel_past_r).at(pel_past_c);
                    G_bac = pel_past_G.at(pel_past_r).at(pel_past_c);
                    B_bac = pel_past_B.at(pel_past_r).at(pel_past_c);
//                    pel_R.at(pel_r).at(pel_c) = pel_past_R.at(pel_past_r).at(pel_past_c);
//                    pel_G.at(pel_r).at(pel_c) = pel_past_G.at(pel_past_r).at(pel_past_c);
//                    pel_B.at(pel_r).at(pel_c) = pel_past_B.at(pel_past_r).at(pel_past_c);
                } else if ( ! right_half_bac && down_half_bac ) {
                    // pel[i][j] = ( pel_past[i+down_for][j+right_for] + pel_past[i+down_for+1][j+right_for] ) // 2;
                    R_bac = round((pel_past_R.at(pel_past_r).at(pel_past_c) + pel_past_R.at(pel_past_r + 1).at(pel_past_c)) / 2);
                    G_bac = round((pel_past_G.at(pel_past_r).at(pel_past_c) + pel_past_G.at(pel_past_r + 1).at(pel_past_c)) / 2);
                    B_bac = round((pel_past_B.at(pel_past_r).at(pel_past_c) + pel_past_B.at(pel_past_r + 1).at(pel_past_c)) / 2);                    
//                    pel_R.at(pel_r).at(pel_c) = round((pel_past_R.at(pel_past_r).at(pel_past_c) + pel_past_R.at(pel_past_r + 1).at(pel_past_c)) / 2);
//                    pel_G.at(pel_r).at(pel_c) = round((pel_past_G.at(pel_past_r).at(pel_past_c) + pel_past_G.at(pel_past_r + 1).at(pel_past_c)) / 2);
//                    pel_B.at(pel_r).at(pel_c) = round((pel_past_B.at(pel_past_r).at(pel_past_c) + pel_past_B.at(pel_past_r + 1).at(pel_past_c)) / 2);
                } else if ( right_half_bac && ! down_half_bac ) {       
                    // pel[i][j] = ( pel_past[i+down_for][j+right_for] + pel_past[i+down_for][j+right_for+1] ) // 2;
                    R_bac = round((pel_past_R.at(pel_past_r).at(pel_past_c) + pel_past_R.at(pel_past_r).at(pel_past_c + 1)) / 2);
                    G_bac = round((pel_past_G.at(pel_past_r).at(pel_past_c) + pel_past_G.at(pel_past_r).at(pel_past_c + 1)) / 2);
                    B_bac = round((pel_past_B.at(pel_past_r).at(pel_past_c) + pel_past_B.at(pel_past_r).at(pel_past_c + 1)) / 2);                    
//                    pel_R.at(pel_r).at(pel_c) = round((pel_past_R.at(pel_past_r).at(pel_past_c) + pel_past_R.at(pel_past_r).at(pel_past_c + 1)) / 2);
//                    pel_G.at(pel_r).at(pel_c) = round((pel_past_G.at(pel_past_r).at(pel_past_c) + pel_past_G.at(pel_past_r).at(pel_past_c + 1)) / 2);
//                    pel_B.at(pel_r).at(pel_c) = round((pel_past_B.at(pel_past_r).at(pel_past_c) + pel_past_B.at(pel_past_r).at(pel_past_c + 1)) / 2);
                } else if ( right_half_bac && down_half_bac ) {
                    // pel[i][j] = ( pel_past[i+down_for][j+right_for] + pel_past[i+down_for+1][j+right_for] + pel_past[i+down_for][j+right_for+1] + pel_past[i+down_for+1][j+right_for+1] ) // 4;
                    R_bac = round((pel_past_R.at(pel_past_r).at(pel_past_c) + pel_past_R.at(pel_past_r + 1).at(pel_past_c) + pel_past_R.at(pel_past_r).at(pel_past_c + 1) + pel_past_R.at(pel_past_r + 1).at(pel_past_c + 1)) / 4);
                    G_bac = round((pel_past_G.at(pel_past_r).at(pel_past_c) + pel_past_G.at(pel_past_r + 1).at(pel_past_c) + pel_past_G.at(pel_past_r).at(pel_past_c + 1) + pel_past_G.at(pel_past_r + 1).at(pel_past_c + 1)) / 4);
                    B_bac = round((pel_past_B.at(pel_past_r).at(pel_past_c) + pel_past_B.at(pel_past_r + 1).at(pel_past_c) + pel_past_B.at(pel_past_r).at(pel_past_c + 1) + pel_past_B.at(pel_past_r + 1).at(pel_past_c + 1)) / 4);
//                    pel_R.at(pel_r).at(pel_c) = round((pel_past_R.at(pel_past_r).at(pel_past_c) + pel_past_R.at(pel_past_r + 1).at(pel_past_c) + pel_past_R.at(pel_past_r).at(pel_past_c + 1) + pel_past_R.at(pel_past_r + 1).at(pel_past_c + 1)) / 4);
//                    pel_G.at(pel_r).at(pel_c) = round((pel_past_G.at(pel_past_r).at(pel_past_c) + pel_past_G.at(pel_past_r + 1).at(pel_past_c) + pel_past_G.at(pel_past_r).at(pel_past_c + 1) + pel_past_G.at(pel_past_r + 1).at(pel_past_c + 1)) / 4);
//                    pel_B.at(pel_r).at(pel_c) = round((pel_past_B.at(pel_past_r).at(pel_past_c) + pel_past_B.at(pel_past_r + 1).at(pel_past_c) + pel_past_B.at(pel_past_r).at(pel_past_c + 1) + pel_past_B.at(pel_past_r + 1).at(pel_past_c + 1)) / 4);                    
                }
            }
            if ((mb_motion_forward == "1") && (mb_motion_backward == "0")) {
                pel_R.at(pel_r).at(pel_c) = R_for;
                pel_G.at(pel_r).at(pel_c) = G_for;
                pel_B.at(pel_r).at(pel_c) = B_for;
            } else if ((mb_motion_forward == "0") && (mb_motion_backward == "1")) {
                pel_R.at(pel_r).at(pel_c) = R_bac;
                pel_G.at(pel_r).at(pel_c) = G_bac;
                pel_B.at(pel_r).at(pel_c) = B_bac;
            } else {
                pel_R.at(pel_r).at(pel_c) = round((R_for + R_bac) / 2);
                pel_G.at(pel_r).at(pel_c) = round((G_for + G_bac) / 2);
                pel_B.at(pel_r).at(pel_c) = round((B_for + B_bac) / 2);
            }            
        }
    }
}




