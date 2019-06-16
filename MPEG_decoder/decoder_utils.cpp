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
Decoder::Decoder() : 
    dct_zz(64, 0), pattern_code(6, 0), dct_recon(8, vector<int>(8, 0)),
    pel_past_R(vector<vector<int>>(240, vector<int>(320, 0))),
    pel_past_G(vector<vector<int>>(240, vector<int>(320, 0))),
    pel_past_B(vector<vector<int>>(240, vector<int>(320, 0))),
    pel_future_R(vector<vector<int>>(240, vector<int>(320, 0))),
    pel_future_G(vector<vector<int>>(240, vector<int>(320, 0))),
    pel_future_B(vector<vector<int>>(240, vector<int>(320, 0))),
    pel_R(vector<vector<int>>(240, vector<int>(320, 0))),
    pel_G(vector<vector<int>>(240, vector<int>(320, 0))),
    pel_B(vector<vector<int>>(240, vector<int>(320, 0))),
    image(240, 320, CV_8UC3),
    y_tmp(vector<vector<int>>(16, vector<int>(16, 0))),
    cb_tmp(vector<vector<int>>(8, vector<int>(8, 0))),
    cr_tmp(vector<vector<int>>(8, vector<int>(8, 0))) {
    // new fast idct
    iclp = iclip+512;
    for (int i= -512; i<512; i++) {
        iclp[i] = (i<-256) ? -256 : ((i>255) ? 255 : i);
    }
}

// Main functions
void Decoder::read_data(string filename) {
    ifstream f (filename, ios::binary);
    if (!f) {
        cout << "File not found !" << endl;
        exit(1);
    }
    que_buf = {istreambuf_iterator<char>{f}, {}};
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
    // bool isend = is_next_start_code(0xb7);
    // if (isend) {
        // cout << "Sequence End Code" << endl;
    // }
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
    // init y, cb, cr final
    pic_num += 1;
    picture_start_code = read_bits(32);
    temporal_reference = read_bits(10);
    picture_coding_type = read_bits(3);
    vbv_delay = read_bits(16);

    if (picture_coding_type == 2 || picture_coding_type == 3) {
        full_pel_forward_vector = read_bits(1);
        forward_f_code = read_bits(3);
        // decode forward_f_code
        forward_r_size = forward_f_code - 1;
        forward_f = 1 << forward_r_size;
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

    if (picture_coding_type != 3) {
        pel_future_R = pel_past_R;
        pel_future_G = pel_past_G;
        pel_future_B = pel_past_B;
    }
    next_start_code();
    if (is_next_slice_code()) {
        // Slice start code '00000101' to '000001AF'
        do {
            slice();
        } while (is_next_slice_code());
    }
    // Reconstruct I-frame
    if (picture_coding_type == 3) {
        // push to imageQueue
        imageQueue.push_back(image);
    } else {
        // update rgb data to pel_past_R, G, B
        for (int r = 0; r< 240; r++) {
            for (int c = 0; c< 320; c++) {
                pel_past_B.at(r).at(c) = image.at<Vec3b>(r, c)[0];
                pel_past_G.at(r).at(c) = image.at<Vec3b>(r, c)[1];
                pel_past_R.at(r).at(c) = image.at<Vec3b>(r, c)[2];
            }
        }
        if (pic_num > 1) {
            // push to img buffer
            rgb2cvmat();
        }
    }
    if (imageQueue.size() > 0) {
        imshow("test", imageQueue.at(0));
        // output_img();
        imageQueue.pop_front();
        waitKey(1);
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
    extra_bit_slice = read_bits(1);
    do {
        macroblock();
    } while (nextbits(23) != 0);
    next_start_code();
}

void Decoder::macroblock() {
    mb_intra = "0";
    // init tmp
    y_tmp = vector<vector<int>>(16, vector<int>(16, 0));
    cb_tmp = vector<vector<int>>(8, vector<int>(8, 0));
    cr_tmp = vector<vector<int>>(8, vector<int>(8, 0));
    // init pattern_code
    pattern_code = {0, 0, 0, 0, 0, 0};
    // init cbp
    cbp = 0;
    // read macroblock stuffing & escape
    uint16_t nbs11 = nextbits(11);
    while (nbs11 == 0xF) {
        mb_stuffing = read_bits(11);
    }
    // read macroblock increment number
    int mb_inc = 0;
    int inc_acc = 0;
    do {
        mb_inc = get_mb_address_map_s();
        if (mb_inc == 101) {
            inc_acc += 33;
        } else {
            inc_acc += mb_inc;
        }
    } while (mb_inc == 101);

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
            recon_image();
        } else if (picture_coding_type == 3) {
            decode_mv();
            recon_image();
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
    // reset dct_dc_past to 1024 when skipped & mb_intra == "0"
    if ((inc_acc > 1) || (mb_intra == "0")) {
        dct_dc_y_past = 1024;
        dct_dc_cb_past = 1024;
        dct_dc_cr_past = 1024;
    }
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
    }
    if (mb_motion_forward == "1") {
        // horizontal
        motion_horizontal_forward_code = get_motion_vector_map_s();
        if (((forward_f) != 1) && (motion_horizontal_forward_code != 0)) {
            motion_horizontal_forward_r = read_bits(forward_r_size);
        }
        // vertical
        motion_vertical_forward_code = get_motion_vector_map_s();
        if (((forward_f) != 1) && (motion_vertical_forward_code != 0)) {
            motion_vertical_forward_r = read_bits(forward_r_size);
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
        motion_horizontal_backward_code = get_motion_vector_map_s();
        if (((backward_f) != 1) && (motion_horizontal_backward_code != 0)) {
            motion_horizontal_backward_r = read_bits(backward_r_size);
        }
        // vertical
        motion_vertical_backward_code = get_motion_vector_map_s();
        if (((backward_f) != 1) && (motion_vertical_backward_code != 0)) {
            motion_vertical_backward_r = read_bits(backward_r_size);
        }
        // motion vectors
        cal_motion_vector_b();
    }
    // init dct dc past
    if (mb_intra == "0") {
        decode_mv();
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
    // recon_image
    recon_image();
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
        if (mb_intra == "1") {
            if (i<4) {
                dct_dc_size_luminance = get_dct_dc_size_lum_map_s();
                if(dct_dc_size_luminance != 0) {
                    dct_dc_differential = read_bits(dct_dc_size_luminance);   
                    if (dct_dc_differential & (1 << (dct_dc_size_luminance-1))) {
                        dct_zz.at(0) = dct_dc_differential;
                    } else {
                        dct_zz.at(0) = (-1 << (dct_dc_size_luminance)) | (dct_dc_differential+1);
                    }
                } else {
                    dct_zz.at(0) = 0;                  
                }
            } else {
                dct_dc_size_chrominance = get_dct_dc_size_chr_map_s();
                if(dct_dc_size_chrominance !=0) {
                    dct_dc_differential = read_bits(dct_dc_size_chrominance);
                    if (dct_dc_differential & (1 << (dct_dc_size_chrominance-1))) {
                        dct_zz.at(0) = dct_dc_differential;
                    } else {
                        dct_zz.at(0) = (-1 << (dct_dc_size_chrominance)) | (dct_dc_differential+1) ;
                    }
                } else {
                    dct_zz.at(0) = 0; 
                }
            }
        } else {
            dct_coeff_first_s();
        }
        if (picture_coding_type != 4) {
            if (mb_intra == "1") {
                dct_zz_i = 0;
            }
            while (nextbits(2) != 2) {
//                dct_coeff_next_s();
//                dct_coeff_next_s2();
                dct_coeff_next_s3();
            }
            end_of_block = read_bits(2);
            // Reconstruct dct_recon
            reconstruct_dct(i);
            // IDCT
            fast_idct();
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

void Decoder::output_img() {
    // Declare output img class
    BMP Bmp_image;
    Bmp_image.SetSize(320, 240);
    for (int r = 0; r < 240; r++) {
        for (int c = 0; c < 320; c++) {
            Bmp_image (c, r) -> Red = imageQueue.at(0).at<Vec3b>(r, c)[2];
            Bmp_image (c, r) -> Green = imageQueue.at(0).at<Vec3b>(r, c)[1];
            Bmp_image (c, r) -> Blue = imageQueue.at(0).at<Vec3b>(r, c)[0];
            Bmp_image (c, r) -> Alpha = 0;
        }
    }
    string wfilename = "C:\\Users\\o1r2g\\OneDrive\\Desktop\\cpp_tutorial\\CPPWorkspace\\MPEG1_S\\pictures\\MPEG_" + to_string(pic_num) + ".bmp";
    Bmp_image.WriteToFile(wfilename.c_str());
}

void Decoder::output_img_mat() {
    // Declare output img class
    BMP Bmp_image;
    Bmp_image.SetSize(320, 240);
    for (int r = 0; r < 240; r++) {
        for (int c = 0; c < 320; c++) {
            Bmp_image (c, r) -> Red = image.at<Vec3b>(r, c)[2];
            Bmp_image (c, r) -> Green = image.at<Vec3b>(r, c)[1];
            Bmp_image (c, r) -> Blue = image.at<Vec3b>(r, c)[0];
            Bmp_image (c, r) -> Alpha = 0;
        }
    }
    string wfilename = "C:\\Users\\o1r2g\\OneDrive\\Desktop\\cpp_tutorial\\CPPWorkspace\\MPEG1_S\\pictures\\MPEG_" + to_string(pic_num) + ".bmp";
    Bmp_image.WriteToFile(wfilename.c_str());
}

// ---> check start & end code
void Decoder::next_start_code() {
    uint32_t next_buf = 0;
    for (int i=0; i<4; i++) {
        next_buf = (next_buf << 8) + (que_buf.at(i));
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
    // Head
    int i = 0;
    int total_rem = num;
    int head_rem = 8 - buf_cursor;
    if (num <= head_rem) {
        // update total_rem
        total_rem = 0;
        if (buf_cursor == 0) {
            // update buffer
            uint8_t tmp_buf = que_buf.at(i);
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
            bit32 = (bit32 << 8) + que_buf.at(i);
            i += 1;
        } else {
            // update total_rem
            total_rem -= head_rem;
            // read all buf in
            uint8_t tmp_buf = buf << buf_cursor;
            bit32 = (bit32 << head_rem) + (tmp_buf >> buf_cursor);
        }
    }
    // Middle
    int quotient = total_rem / 8;

    for (int j=0; j<quotient; i++, j++) {
        // update total_rem
        total_rem -= 8;
        // read 8-bit * N
        bit32 = (bit32 << 8) + que_buf.at(i);
    }
    // Last
    if (total_rem > 0) {
        // update buffer
        uint8_t tmp_buf = que_buf.at(i);
        bit32 = (bit32 << total_rem) + (tmp_buf >> (8 - total_rem));
    }
    return bit32;  
}

uint32_t Decoder::read_bits(int num) {
    uint32_t bit32  = 0;
    int new_cur_pos = get_cur_pos(buf_cursor, num);
    // Head
    int total_rem = num;
    int head_rem = 8 - buf_cursor;
    if (num <= head_rem) {
        // update total_rem
        total_rem = 0;        
        if (buf_cursor == 0) {
            // update buffer
            buf = que_buf.at(0);
            bit32 = (bit32 << num) + (buf >> (8 - num));
            que_buf.pop_front();
        } else {
            uint8_t tmp_buf = buf << buf_cursor;
            bit32 = (bit32 << num) + (tmp_buf >> (8 - (buf_cursor + num) + buf_cursor));
        }
    } else {
        if (buf_cursor == 0) {
            // update total_rem
            total_rem -= 8;
            // read 8-bit * 1
            bit32 = (bit32 << 8) + que_buf.at(0);
            que_buf.pop_front();
        } else {
            // update total_rem
            total_rem -= head_rem;
            // read all buf in
            uint8_t tmp_buf = buf << buf_cursor;
            bit32 = (bit32 << head_rem) + (tmp_buf >> buf_cursor);
        }
    }
    // Middle
    int quotient = total_rem / 8;

    for (int i=0; i<quotient; i++) {
        // update total_rem
        total_rem -= 8;
        // read 8-bit * N
        bit32 = (bit32 << 8) + que_buf.at(0);
        que_buf.pop_front();
    }
    // Last
    if (total_rem > 0) {
        // update buffer
        buf = que_buf.at(0);
        bit32 = (bit32 << total_rem) + (buf >> (8 - total_rem));
        que_buf.pop_front();
    }
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
    }
}

// Get Mapping Value
int Decoder::get_mb_address_map_s() {
    uint32_t nbs32 = 0;
    uint16_t second_nbs16 = 0;
    for (int i=1; i<12; i++) {
        second_nbs16 = nextbits(i);
        nbs32 = i;
        nbs32 = (nbs32 << 16) + second_nbs16;
        // Lookup table via switch optimization
        uint8_t _ = 0;
        switch (nbs32) {
            case 0xb0008:
                _ = read_bits(i);
                i = 12;
                return 101;
            case 0xb000f:
                _ = read_bits(i);
                i = 12;
                return 100;
            case 0xb0018:
                _ = read_bits(i);
                i = 12;
                return 33;
            case 0xb0019:
                _ = read_bits(i);
                i = 12;
                return 32;
            case 0xb001a:
                _ = read_bits(i);
                i = 12;
                return 31;
            case 0xb001b:
                _ = read_bits(i);
                i = 12;
                return 30;
            case 0xb001c:
                _ = read_bits(i);
                i = 12;
                return 29;
            case 0xb001d:
                _ = read_bits(i);
                i = 12;
                return 28;
            case 0xb001e:
                _ = read_bits(i);
                i = 12;
                return 27;
            case 0xb001f:
                _ = read_bits(i);
                i = 12;
                return 26;
            case 0xb0020:
                _ = read_bits(i);
                i = 12;
                return 25;
            case 0xb0021:
                _ = read_bits(i);
                i = 12;
                return 24;
            case 0xb0022:
                _ = read_bits(i);
                i = 12;
                return 23;
            case 0xb0023:
                _ = read_bits(i);
                i = 12;
                return 22;
            case 0xa0012:
                _ = read_bits(i);
                i = 12;
                return 21;
            case 0xa0013:
                _ = read_bits(i);
                i = 12;
                return 20;
            case 0xa0014:
                _ = read_bits(i);
                i = 12;
                return 19;
            case 0xa0015:
                _ = read_bits(i);
                i = 12;
                return 18;
            case 0xa0016:
                _ = read_bits(i);
                i = 12;
                return 17;
            case 0xa0017:
                _ = read_bits(i);
                i = 12;
                return 16;
            case 0x80006:
                _ = read_bits(i);
                i = 12;
                return 15;
            case 0x80007:
                _ = read_bits(i);
                i = 12;
                return 14;
            case 0x80008:
                _ = read_bits(i);
                i = 12;
                return 13;
            case 0x80009:
                _ = read_bits(i);
                i = 12;
                return 12;
            case 0x8000a:
                _ = read_bits(i);
                i = 12;
                return 11;
            case 0x8000b:
                _ = read_bits(i);
                i = 12;
                return 10;
            case 0x70006:
                _ = read_bits(i);
                i = 12;
                return 9;
            case 0x70007:
                _ = read_bits(i);
                i = 12;
                return 8;
            case 0x50002:
                _ = read_bits(i);
                i = 12;
                return 7;
            case 0x50003:
                _ = read_bits(i);
                i = 12;
                return 6;
            case 0x40002:
                _ = read_bits(i);
                i = 12;
                return 5;
            case 0x40003:
                _ = read_bits(i);
                i = 12;
                return 4;
            case 0x30002:
                _ = read_bits(i);
                i = 12;
                return 3;
            case 0x30003:
                _ = read_bits(i);
                i = 12;
                return 2;
            case 0x10001:
                _ = read_bits(i);
                i = 12;
                return 1;
        }
    }
}

string Decoder::get_mb_type_map() {
    uint8_t nbs6 = nextbits(6);
    uint8_t tmp_nbs = 0;
    for (int i=1; i<7; i++) {
        string s = bitset<6>(nbs6).to_string();
        s = s.substr(0, i);
        map<string, string>::iterator iter;
        if (picture_coding_type == 1) {
            // i-frame
            iter = mb_type_i_map.find(s);
            if (iter !=mb_type_i_map.end()) {
                    uint8_t _ = read_bits(i);
                    return iter->second;
            }         
        } else if (picture_coding_type == 2) {
            // p-frame
            iter = mb_type_p_map.find(s);
            if (iter !=mb_type_p_map.end()) {
                    uint8_t _ = read_bits(i);
                    return iter->second;
            }          
        } else if (picture_coding_type == 3) {
            // b-frame
            iter = mb_type_b_map.find(s);
            if (iter !=mb_type_b_map.end()) {
                    uint8_t _ = read_bits(i);
                    return iter->second;
            }      
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
        map<string, int>::iterator iter;
        iter = motion_vector_map.find(s);
        if (iter !=motion_vector_map.end()) {
            uint8_t _ = read_bits(i);
            return iter->second;
        }
    }
}

int Decoder::get_motion_vector_map_s() {
    uint16_t nbs16 = 0;
    for (int i=1; i<12; i++) {
        nbs16 = i;
        nbs16 = (nbs16 << 6) + nextbits(i);
        // Lookup table via switch optimization
        uint8_t _ = 0;
        switch (nbs16) {
            case 0x2d8:
                _ = read_bits(i);
                i = 12;
                return 16;
            case 0x2d9:
                _ = read_bits(i);
                i = 12;
                return -16;
            case 0x2da:
                _ = read_bits(i);
                i = 12;
                return 15;
            case 0x2db:
                _ = read_bits(i);
                i = 12;
                return -15;
            case 0x2dc:
                _ = read_bits(i);
                i = 12;
                return 14;
            case 0x2dd:
                _ = read_bits(i);
                i = 12;
                return -14;
            case 0x2de:
                _ = read_bits(i);
                i = 12;
                return 13;
            case 0x2df:
                _ = read_bits(i);
                i = 12;
                return -13;
            case 0x2e0:
                _ = read_bits(i);
                i = 12;
                return 12;
            case 0x2e1:
                _ = read_bits(i);
                i = 12;
                return -12;
            case 0x2e2:
                _ = read_bits(i);
                i = 12;
                return 11;
            case 0x2e3:
                _ = read_bits(i);
                i = 12;
                return -11;
            case 0x292:
                _ = read_bits(i);
                i = 12;
                return 10;
            case 0x293:
                _ = read_bits(i);
                i = 12;
                return -10;
            case 0x294:
                _ = read_bits(i);
                i = 12;
                return 9;
            case 0x295:
                _ = read_bits(i);
                i = 12;
                return -9;
            case 0x296:
                _ = read_bits(i);
                i = 12;
                return 8;
            case 0x297:
                _ = read_bits(i);
                i = 12;
                return -8;
            case 0x206:
                _ = read_bits(i);
                i = 12;
                return 7;
            case 0x207:
                _ = read_bits(i);
                i = 12;
                return -7;
            case 0x208:
                _ = read_bits(i);
                i = 12;
                return 6;
            case 0x209:
                _ = read_bits(i);
                i = 12;
                return -6;
            case 0x20a:
                _ = read_bits(i);
                i = 12;
                return 5;
            case 0x20b:
                _ = read_bits(i);
                i = 12;
                return -5;
            case 0x1c6:
                _ = read_bits(i);
                i = 12;
                return 4;
            case 0x1c7:
                _ = read_bits(i);
                i = 12;
                return -4;
            case 0x142:
                _ = read_bits(i);
                i = 12;
                return 3;
            case 0x143:
                _ = read_bits(i);
                i = 12;
                return -3;
            case 0x102:
                _ = read_bits(i);
                i = 12;
                return 2;
            case 0x103:
                _ = read_bits(i);
                i = 12;
                return -2;
            case 0xc2:
                _ = read_bits(i);
                i = 12;
                return 1;
            case 0xc3:
                _ = read_bits(i);
                i = 12;
                return -1;
            case 0x41:
                _ = read_bits(i);
                i = 12;
                return 0;
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

void Decoder::dct_coeff_next_s2() {
    uint32_t tmp;
    uint64_t nbs64 = 0;
    uint32_t second_nbs32 = nextbits(29);
    for (int i=2; i<29; i++) {
        nbs64 = i;
        nbs64 = (nbs64 << 32) + (second_nbs32 >> (29 - i));
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

void Decoder::dct_coeff_next_s3() {
    uint32_t nbs32 = 0;
    for (int i=2; i<29; i++) {
        nbs32 = i;
        nbs32 = (nbs32 << 6) + nextbits(i);
        // Lookup table via switch optimization
        uint8_t _ = 0;
        switch (nbs32) {
            case 0x410:
                _ = read_bits(i);
                fill_dct_zz(1, 18);
                i = 29;
                break;
            case 0x411:
                _ = read_bits(i);
                fill_dct_zz(1, 17);
                i = 29;
                break;
            case 0x412:
                _ = read_bits(i);
                fill_dct_zz(1, 16);
                i = 29;
                break;
            case 0x413:
                _ = read_bits(i);
                fill_dct_zz(1, 15);
                i = 29;
                break;
            case 0x414:
                _ = read_bits(i);
                fill_dct_zz(6, 3);
                i = 29;
                break;
            case 0x415:
                _ = read_bits(i);
                fill_dct_zz(16, 2);
                i = 29;
                break;
            case 0x416:
                _ = read_bits(i);
                fill_dct_zz(15, 2);
                i = 29;
                break;
            case 0x417:
                _ = read_bits(i);
                fill_dct_zz(14, 2);
                i = 29;
                break;
            case 0x418:
                _ = read_bits(i);
                fill_dct_zz(13, 2);
                i = 29;
                break;
            case 0x419:
                _ = read_bits(i);
                fill_dct_zz(12, 2);
                i = 29;
                break;
            case 0x41a:
                _ = read_bits(i);
                fill_dct_zz(11, 2);
                i = 29;
                break;
            case 0x41b:
                _ = read_bits(i);
                fill_dct_zz(31, 1);
                i = 29;
                break;
            case 0x41c:
                _ = read_bits(i);
                fill_dct_zz(30, 1);
                i = 29;
                break;
            case 0x41d:
                _ = read_bits(i);
                fill_dct_zz(29, 1);
                i = 29;
                break;
            case 0x41e:
                _ = read_bits(i);
                fill_dct_zz(28, 1);
                i = 29;
                break;
            case 0x41f:
                _ = read_bits(i);
                fill_dct_zz(27, 1);
                i = 29;
                break;
            case 0x3d0:
                _ = read_bits(i);
                fill_dct_zz(0, 40);
                i = 29;
                break;
            case 0x3d1:
                _ = read_bits(i);
                fill_dct_zz(0, 39);
                i = 29;
                break;
            case 0x3d2:
                _ = read_bits(i);
                fill_dct_zz(0, 38);
                i = 29;
                break;
            case 0x3d3:
                _ = read_bits(i);
                fill_dct_zz(0, 37);
                i = 29;
                break;
            case 0x3d4:
                _ = read_bits(i);
                fill_dct_zz(0, 36);
                i = 29;
                break;
            case 0x3d5:
                _ = read_bits(i);
                fill_dct_zz(0, 35);
                i = 29;
                break;
            case 0x3d6:
                _ = read_bits(i);
                fill_dct_zz(0, 34);
                i = 29;
                break;
            case 0x3d7:
                _ = read_bits(i);
                fill_dct_zz(0, 33);
                i = 29;
                break;
            case 0x3d8:
                _ = read_bits(i);
                fill_dct_zz(0, 32);
                i = 29;
                break;
            case 0x3d9:
                _ = read_bits(i);
                fill_dct_zz(1, 14);
                i = 29;
                break;
            case 0x3da:
                _ = read_bits(i);
                fill_dct_zz(1, 13);
                i = 29;
                break;
            case 0x3db:
                _ = read_bits(i);
                fill_dct_zz(1, 12);
                i = 29;
                break;
            case 0x3dc:
                _ = read_bits(i);
                fill_dct_zz(1, 11);
                i = 29;
                break;
            case 0x3dd:
                _ = read_bits(i);
                fill_dct_zz(1, 10);
                i = 29;
                break;
            case 0x3de:
                _ = read_bits(i);
                fill_dct_zz(1, 9);
                i = 29;
                break;
            case 0x3df:
                _ = read_bits(i);
                fill_dct_zz(1, 8);
                i = 29;
                break;
            case 0x390:
                _ = read_bits(i);
                fill_dct_zz(0, 31);
                i = 29;
                break;
            case 0x391:
                _ = read_bits(i);
                fill_dct_zz(0, 30);
                i = 29;
                break;
            case 0x392:
                _ = read_bits(i);
                fill_dct_zz(0, 29);
                i = 29;
                break;
            case 0x393:
                _ = read_bits(i);
                fill_dct_zz(0, 28);
                i = 29;
                break;
            case 0x394:
                _ = read_bits(i);
                fill_dct_zz(0, 27);
                i = 29;
                break;
            case 0x395:
                _ = read_bits(i);
                fill_dct_zz(0, 26);
                i = 29;
                break;
            case 0x396:
                _ = read_bits(i);
                fill_dct_zz(0, 25);
                i = 29;
                break;
            case 0x397:
                _ = read_bits(i);
                fill_dct_zz(0, 24);
                i = 29;
                break;
            case 0x398:
                _ = read_bits(i);
                fill_dct_zz(0, 23);
                i = 29;
                break;
            case 0x399:
                _ = read_bits(i);
                fill_dct_zz(0, 22);
                i = 29;
                break;
            case 0x39a:
                _ = read_bits(i);
                fill_dct_zz(0, 21);
                i = 29;
                break;
            case 0x39b:
                _ = read_bits(i);
                fill_dct_zz(0, 20);
                i = 29;
                break;
            case 0x39c:
                _ = read_bits(i);
                fill_dct_zz(0, 19);
                i = 29;
                break;
            case 0x39d:
                _ = read_bits(i);
                fill_dct_zz(0, 18);
                i = 29;
                break;
            case 0x39e:
                _ = read_bits(i);
                fill_dct_zz(0, 17);
                i = 29;
                break;
            case 0x39f:
                _ = read_bits(i);
                fill_dct_zz(0, 16);
                i = 29;
                break;
            case 0x350:
                _ = read_bits(i);
                fill_dct_zz(10, 2);
                i = 29;
                break;
            case 0x351:
                _ = read_bits(i);
                fill_dct_zz(9, 2);
                i = 29;
                break;
            case 0x352:
                _ = read_bits(i);
                fill_dct_zz(5, 3);
                i = 29;
                break;
            case 0x353:
                _ = read_bits(i);
                fill_dct_zz(3, 4);
                i = 29;
                break;
            case 0x354:
                _ = read_bits(i);
                fill_dct_zz(2, 5);
                i = 29;
                break;
            case 0x355:
                _ = read_bits(i);
                fill_dct_zz(1, 7);
                i = 29;
                break;
            case 0x356:
                _ = read_bits(i);
                fill_dct_zz(1, 6);
                i = 29;
                break;
            case 0x357:
                _ = read_bits(i);
                fill_dct_zz(0, 15);
                i = 29;
                break;
            case 0x358:
                _ = read_bits(i);
                fill_dct_zz(0, 14);
                i = 29;
                break;
            case 0x359:
                _ = read_bits(i);
                fill_dct_zz(0, 13);
                i = 29;
                break;
            case 0x35a:
                _ = read_bits(i);
                fill_dct_zz(0, 12);
                i = 29;
                break;
            case 0x35b:
                _ = read_bits(i);
                fill_dct_zz(26, 1);
                i = 29;
                break;
            case 0x35c:
                _ = read_bits(i);
                fill_dct_zz(25, 1);
                i = 29;
                break;
            case 0x35d:
                _ = read_bits(i);
                fill_dct_zz(24, 1);
                i = 29;
                break;
            case 0x35e:
                _ = read_bits(i);
                fill_dct_zz(23, 1);
                i = 29;
                break;
            case 0x35f:
                _ = read_bits(i);
                fill_dct_zz(22, 1);
                i = 29;
                break;
            case 0x310:
                _ = read_bits(i);
                fill_dct_zz(0, 11);
                i = 29;
                break;
            case 0x311:
                _ = read_bits(i);
                fill_dct_zz(8, 2);
                i = 29;
                break;
            case 0x312:
                _ = read_bits(i);
                fill_dct_zz(4, 3);
                i = 29;
                break;
            case 0x313:
                _ = read_bits(i);
                fill_dct_zz(0, 10);
                i = 29;
                break;
            case 0x314:
                _ = read_bits(i);
                fill_dct_zz(2, 4);
                i = 29;
                break;
            case 0x315:
                _ = read_bits(i);
                fill_dct_zz(7, 2);
                i = 29;
                break;
            case 0x316:
                _ = read_bits(i);
                fill_dct_zz(21, 1);
                i = 29;
                break;
            case 0x317:
                _ = read_bits(i);
                fill_dct_zz(20, 1);
                i = 29;
                break;
            case 0x318:
                _ = read_bits(i);
                fill_dct_zz(0, 9);
                i = 29;
                break;
            case 0x319:
                _ = read_bits(i);
                fill_dct_zz(19, 1);
                i = 29;
                break;
            case 0x31a:
                _ = read_bits(i);
                fill_dct_zz(18, 1);
                i = 29;
                break;
            case 0x31b:
                _ = read_bits(i);
                fill_dct_zz(1, 5);
                i = 29;
                break;
            case 0x31c:
                _ = read_bits(i);
                fill_dct_zz(3, 3);
                i = 29;
                break;
            case 0x31d:
                _ = read_bits(i);
                fill_dct_zz(0, 8);
                i = 29;
                break;
            case 0x31e:
                _ = read_bits(i);
                fill_dct_zz(6, 2);
                i = 29;
                break;
            case 0x31f:
                _ = read_bits(i);
                fill_dct_zz(17, 1);
                i = 29;
                break;
            case 0x288:
                _ = read_bits(i);
                fill_dct_zz(16, 1);
                i = 29;
                break;
            case 0x289:
                _ = read_bits(i);
                fill_dct_zz(5, 2);
                i = 29;
                break;
            case 0x28a:
                _ = read_bits(i);
                fill_dct_zz(0, 7);
                i = 29;
                break;
            case 0x28b:
                _ = read_bits(i);
                fill_dct_zz(2, 3);
                i = 29;
                break;
            case 0x28c:
                _ = read_bits(i);
                fill_dct_zz(1, 4);
                i = 29;
                break;
            case 0x28d:
                _ = read_bits(i);
                fill_dct_zz(15, 1);
                i = 29;
                break;
            case 0x28e:
                _ = read_bits(i);
                fill_dct_zz(14, 1);
                i = 29;
                break;
            case 0x28f:
                _ = read_bits(i);
                fill_dct_zz(4, 2);
                i = 29;
                break;
            case 0x181:
                _ = read_bits(i);
                fill_dct_zz(-1, -1);
                i = 29;
                break;
            case 0x1c4:
                _ = read_bits(i);
                fill_dct_zz(2, 2);
                i = 29;
                break;
            case 0x1c5:
                _ = read_bits(i);
                fill_dct_zz(9, 1);
                i = 29;
                break;
            case 0x1c6:
                _ = read_bits(i);
                fill_dct_zz(0, 4);
                i = 29;
                break;
            case 0x1c7:
                _ = read_bits(i);
                fill_dct_zz(8, 1);
                i = 29;
                break;
            case 0x184:
                _ = read_bits(i);
                fill_dct_zz(7, 1);
                i = 29;
                break;
            case 0x185:
                _ = read_bits(i);
                fill_dct_zz(6, 1);
                i = 29;
                break;
            case 0x186:
                _ = read_bits(i);
                fill_dct_zz(1, 2);
                i = 29;
                break;
            case 0x187:
                _ = read_bits(i);
                fill_dct_zz(5, 1);
                i = 29;
                break;
            case 0x220:
                _ = read_bits(i);
                fill_dct_zz(13, 1);
                i = 29;
                break;
            case 0x221:
                _ = read_bits(i);
                fill_dct_zz(0, 6);
                i = 29;
                break;
            case 0x222:
                _ = read_bits(i);
                fill_dct_zz(12, 1);
                i = 29;
                break;
            case 0x223:
                _ = read_bits(i);
                fill_dct_zz(11, 1);
                i = 29;
                break;
            case 0x224:
                _ = read_bits(i);
                fill_dct_zz(3, 2);
                i = 29;
                break;
            case 0x225:
                _ = read_bits(i);
                fill_dct_zz(1, 3);
                i = 29;
                break;
            case 0x226:
                _ = read_bits(i);
                fill_dct_zz(0, 5);
                i = 29;
                break;
            case 0x227:
                _ = read_bits(i);
                fill_dct_zz(10, 1);
                i = 29;
                break;
            case 0x145:
                _ = read_bits(i);
                fill_dct_zz(0, 3);
                i = 29;
                break;
            case 0x146:
                _ = read_bits(i);
                fill_dct_zz(4, 1);
                i = 29;
                break;
            case 0x147:
                _ = read_bits(i);
                fill_dct_zz(3, 1);
                i = 29;
                break;
            case 0x104:
                _ = read_bits(i);
                fill_dct_zz(0, 2);
                i = 29;
                break;
            case 0x105:
                _ = read_bits(i);
                fill_dct_zz(2, 1);
                i = 29;
                break;
            case 0xc3:
                _ = read_bits(i);
                fill_dct_zz(1, 1);
                i = 29;
                break;
            case 0x83:
                _ = read_bits(i);
                fill_dct_zz(0, 1);
                i = 29;
                break;
        }
    }
}

void Decoder::fill_dct_zz(int run, int level) {
    // normal case
    if (run != -1) {
        dct_zz_i = dct_zz_i + run +1;
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
        dct_zz_i = dct_zz_i + run +1;
        dct_zz.at(dct_zz_i) = level;
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
            uint8_t tmp_lc = ~(last8_code - 1);
            uint16_t tmp = 0;
            tmp += tmp_lc;
            int neg_total_code = - tmp;
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
    if (mb_intra == "1") {
        int i = 0;
        for (int m=0; m<8; m++) {
            for (int n=0; n<8; n++) {
                i = zigzag_m[m][n];
                dct_recon[m][n] = ( 2 * dct_zz[i] * quantizer_scale * intra_quant[m][n] ) / 16;
                // dct_recon[m][n] = (dct_zz[i] * quantizer_scale * intra_quant[m][n]) >> 3;
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
                dct_recon[m][n] = ( ( (2 * dct_zz[i]) + sign(dct_zz[i]) ) * quantizer_scale * non_intra_quant[m][n] ) / 16;
                // dct_recon[m][n] = ((dct_zz[i] + sign(dct_zz[i])) * quantizer_scale * non_intra_quant[m][n] ) * 0.125;
                // dct_recon[m][n] = ((dct_zz[i] + sign(dct_zz[i])) * quantizer_scale * non_intra_quant[m][n] ) >> 3;
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
}

void Decoder::idctrow(int i) {
    // Get each i row from 2d-vector
    vector<int>blk(8, 0);
    int num = 0;
    for (int r=0; r<8; r++) {
        for (int c=0; c<8; c++) {
            if (r == i) {
                blk.at(num) = dct_recon.at(r).at(c);
                num++;
            }
        }
    }

    int x0, x8;
    int x1 = blk[4];
    int x2 = blk[6];
    int x3 = blk[2];
    int x4 = blk[1];
    int x5 = blk[7];
    int x6 = blk[5];
    int x7 = blk[3];
    if (!((x1<<11) | x2 | x3 | x4 | x5 | x6 | x7 )) {
        dct_recon[i][0]=dct_recon[i][1]=dct_recon[i][2]=dct_recon[i][3]=dct_recon[i][4]=dct_recon[i][5]=dct_recon[i][6]=dct_recon[i][7]=blk[0]<<3;
        return;
    }
    x0 = (blk[0]<<11) + 128;

    // First stage
    x8 = W7*(x4+x5);
    x4 = x8 + (W1-W7)*x4;
    x5 = x8 - (W1+W7)*x5;
    x8 = W3*(x6+x7);
    x6 = x8 - (W3-W5)*x6;
    x7 = x8 - (W3+W5)*x7;

    // second stage
    x8 = x0 + x1;
    x0 -= x1;
    x1 = W6*(x3+x2);
    x2 = x1 - (W2+W6)*x2;
    x3 = x1 + (W2-W6)*x3;
    x1 = x4 + x6;
    x4 -= x6;
    x6 = x5 + x7;
    x5 -= x7;

    // third stage
    x7 = x8 + x3;
    x8 -= x3;
    x3 = x0 + x2;
    x0 -= x2;
    x2 = (181*(x4+x5)+128)>>8;
    x4 = (181*(x4-x5)+128)>>8;

    // fourth stage : assign value back to dct_recon 2d-vector
    dct_recon[i][0] = (x7+x1)>>8;
    dct_recon[i][1] = (x3+x2)>>8;
    dct_recon[i][2] = (x0+x4)>>8;
    dct_recon[i][3] = (x8+x6)>>8;
    dct_recon[i][4] = (x8-x6)>>8;
    dct_recon[i][5] = (x0-x4)>>8;
    dct_recon[i][6] = (x3-x2)>>8;
    dct_recon[i][7] = (x7-x1)>>8;
}

void Decoder::idctcol(int i) {
    // Get each i col from 2d-vector
    vector<int>blk(8, 0);
    int num = 0;
    for (int r=0; r<8; r++) {
        for (int c=0; c<8; c++) {
            if (c == i) {
                blk.at(num) = dct_recon.at(r).at(c);
                num ++;
            }
        }
    }

    int x0, x8;
    int x1 = (blk[4]<<8);
    int x2 = blk[6];
    int x3 = blk[2];
    int x4 = blk[1];
    int x5 = blk[7];
    int x6 = blk[5];
    int x7 = blk[3];
    if (!(x1 | x2 | x3 | x4 | x5 | x6 | x7)) {
        dct_recon[0][i]=dct_recon[1][i]=dct_recon[2][i]=dct_recon[3][i]=dct_recon[4][i]=dct_recon[5][i]=dct_recon[6][i]=dct_recon[7][i]=iclp[(blk[0]+32)>>6];
        return;
    }
    x0 = (blk[0]<<8) + 8192;

    // first stage
    x8 = W7*(x4+x5) + 4;
    x4 = (x8+(W1-W7)*x4)>>3;
    x5 = (x8-(W1+W7)*x5)>>3;
    x8 = W3*(x6+x7) + 4;
    x6 = (x8-(W3-W5)*x6)>>3;
    x7 = (x8-(W3+W5)*x7)>>3;
    
    // second stage
    x8 = x0 + x1;
    x0 -= x1;
    x1 = W6*(x3+x2) + 4;
    x2 = (x1-(W2+W6)*x2)>>3;
    x3 = (x1+(W2-W6)*x3)>>3;
    x1 = x4 + x6;
    x4 -= x6;
    x6 = x5 + x7;
    x5 -= x7;

    // third stage
    x7 = x8 + x3;
    x8 -= x3;
    x3 = x0 + x2;
    x0 -= x2;
    x2 = (181*(x4+x5)+128)>>8;
    x4 = (181*(x4-x5)+128)>>8;

    // fourth stage : assign value back to dct_recon 2d-vector
    dct_recon[0][i] = iclp[(x7+x1)>>14];
    dct_recon[1][i] = iclp[(x3+x2)>>14];
    dct_recon[2][i] = iclp[(x0+x4)>>14];
    dct_recon[3][i] = iclp[(x8+x6)>>14];
    dct_recon[4][i] = iclp[(x8-x6)>>14];
    dct_recon[5][i] = iclp[(x0-x4)>>14];
    dct_recon[6][i] = iclp[(x3-x2)>>14];
    dct_recon[7][i] = iclp[(x7-x1)>>14];
}

void Decoder::fast_idct() {
    // Reference
    //      Inverse two dimensional DCT, Chen-Wang algorithm
    //      ( cf. IEEE ASSP-32, pp. 803-816, Aug. 1984 )
    //      ( https://github.com/keithw/mympeg2enc/blob/master/idct.c#L58 )
    int i;
    for (i=0; i<8; i++) {
        idctrow(i);
    }
    for (i=0; i<8; i++) {
        idctcol(i);
    }
    switch (picture_coding_type) {
        case 1:
            for (int i=0; i<8; i++) {
                for (int j=0; j<8; j++) {
                    if (dct_recon[i][j] < 0) {
                        dct_recon.at(i).at(j) = 0;
                    } else if (dct_recon[i][j] > 255) {
                        dct_recon.at(i).at(j) = 255;
                    }
                }
            }
            break;
        case 2:
        case 3:
            break;
    }
    collect_mbs();
}

void Decoder::rgb2cvmat() {
    Mat fut_image(240, 320, CV_8UC3);
    for (int r = 0; r< 240; r++) {
        for (int c = 0; c< 320; c++) {
            fut_image.at<Vec3b>(r, c)[0] = pel_future_B[r][c];
            fut_image.at<Vec3b>(r, c)[1] = pel_future_G[r][c];
            fut_image.at<Vec3b>(r, c)[2] = pel_future_R[r][c];
        }
    }
    imageQueue.push_back(fut_image);
}

void Decoder::ycbcr2rgb_s(bool to_buffer, bool to_output) {
    int img_y_w = y_result_final.at(0).size();
    int img_y_h = y_result_final.size();
    double R;
    double G;
    double B;
    if (to_output) {
        Mat image(240, 320, CV_8UC3);
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
                image.at<Vec3b>(r, c)[0] = (int)B;
                image.at<Vec3b>(r, c)[1] = (int)G;
                image.at<Vec3b>(r, c)[2] = (int)R;
            }
        }
        imageQueue.push_back(image);
    }
    if (to_buffer) {
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
                pel_past_R.at(r).at(c) = (int)R;
                pel_past_G.at(r).at(c) = (int)G;
                pel_past_B.at(r).at(c) = (int)B;
            }
        }
    }
}

void Decoder::collect_mbs() {
    int y_r = mb_row * 16;
    int y_c = mb_col * 16;
    int c_r = mb_row * 8;
    int c_c = mb_col * 8;
    switch (block_i) {
        case 0:
            for (int r=0; r<8; r++) {
                for (int c=0; c<8; c++) {
                    // y_result_final.at(y_r + r).at(y_c + c) = dct_recon.at(r).at(c);
                    y_tmp.at(r).at(c) = dct_recon.at(r).at(c);
                }
            }
            break;
        case 1:
            for (int r=0; r<8; r++) {
                for (int c=0; c<8; c++) {
                    // y_result_final.at(y_r + r).at(y_c + 8 + c) = dct_recon.at(r).at(c);
                    y_tmp.at(r).at(c + 8) = dct_recon.at(r).at(c);
                }
            }
            break;
        case 2:
            for (int r=0; r<8; r++) {
                for (int c=0; c<8; c++) {
                    // y_result_final.at(y_r + 8 + r).at(y_c + c) = dct_recon.at(r).at(c);
                    y_tmp.at(r + 8).at(c) = dct_recon.at(r).at(c);
                }
            }
            break;
        case 3:
            for (int r=0; r<8; r++) {
                for (int c=0; c<8; c++) {
                    // y_result_final.at(y_r + 8 + r).at(y_c + 8 + c) = dct_recon.at(r).at(c);
                    y_tmp.at(r + 8).at(c + 8) = dct_recon.at(r).at(c);
                }
            }
            break;
        case 4:
            for (int r=0; r<8; r++) {
                for (int c=0; c<8; c++) {
                    // cb_result_final.at(c_r + r).at(c_c + c) = dct_recon.at(r).at(c);
                    cb_tmp.at(r).at(c) = dct_recon.at(r).at(c);
                }
            }
            break;
        case 5:
            for (int r=0; r<8; r++) {
                for (int c=0; c<8; c++) {
                    // cr_result_final.at(c_r + r).at(c_c + c) = dct_recon.at(r).at(c);
                    cr_tmp.at(r).at(c) = dct_recon.at(r).at(c);
                }
            }
            break;
    }
}

void Decoder::recon_image() {
    int pel_r = mb_row * 16;
    int pel_c = mb_col * 16;
    double R;
    double G;
    double B;
    for (int r=0; r<16; r++) {
        for (int c=0; c<16; c++) {
            // For Cb, Cr
            int c_r = r / 2;
            int c_c = c / 2;
            double Y = y_tmp.at(r).at(c);
            double Cb = cb_tmp.at(c_r).at(c_c);
            double Cr = cr_tmp.at(c_r).at(c_c);
            if (mb_intra == "1") {
                R = Y + (1.402 * (Cr - 128));
                G = Y - (0.344 * (Cb - 128)) - (0.714 * (Cr - 128));
                B = Y + (1.772 * (Cb - 128));
            } else {
                R = pel_R.at(pel_r + r).at(pel_c + c);
                G = pel_G.at(pel_r + r).at(pel_c + c);
                B = pel_B.at(pel_r + r).at(pel_c + c);
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
            image.at<Vec3b>(pel_r + r, pel_c + c)[0] = (int)B;
            image.at<Vec3b>(pel_r + r, pel_c + c)[1] = (int)G;
            image.at<Vec3b>(pel_r + r, pel_c + c)[2] = (int)R;
        }
    }
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
            if (mb_motion_forward == "1") {
                int pel_future_r = pel_r + down_for;
                int pel_future_c = pel_c + right_for;
                if ( ! right_half_for && ! down_half_for ) {
                    // pel[i][j] = pel_past[i+down_for][j+right_for];
                    R_for = pel_future_R.at(pel_future_r).at(pel_future_c);
                    G_for = pel_future_G.at(pel_future_r).at(pel_future_c);
                    B_for = pel_future_B.at(pel_future_r).at(pel_future_c);
                } else if ( ! right_half_for && down_half_for ) {
                    // pel[i][j] = ( pel_past[i+down_for][j+right_for] + pel_past[i+down_for+1][j+right_for] ) // 2;
                    R_for = round((pel_future_R.at(pel_future_r).at(pel_future_c) + pel_future_R.at(pel_future_r + 1).at(pel_future_c)) / 2);
                    G_for = round((pel_future_G.at(pel_future_r).at(pel_future_c) + pel_future_G.at(pel_future_r + 1).at(pel_future_c)) / 2);
                    B_for = round((pel_future_B.at(pel_future_r).at(pel_future_c) + pel_future_B.at(pel_future_r + 1).at(pel_future_c)) / 2);
                } else if ( right_half_for && ! down_half_for ) {
                    // pel[i][j] = ( pel_past[i+down_for][j+right_for] + pel_past[i+down_for][j+right_for+1] ) // 2;
                    R_for = round((pel_future_R.at(pel_future_r).at(pel_future_c) + pel_future_R.at(pel_future_r).at(pel_future_c + 1)) / 2);
                    G_for = round((pel_future_G.at(pel_future_r).at(pel_future_c) + pel_future_G.at(pel_future_r).at(pel_future_c + 1)) / 2);
                    B_for = round((pel_future_B.at(pel_future_r).at(pel_future_c) + pel_future_B.at(pel_future_r).at(pel_future_c + 1)) / 2);
                } else if ( right_half_for && down_half_for ) {
                    // pel[i][j] = ( pel_past[i+down_for][j+right_for] + pel_past[i+down_for+1][j+right_for] + pel_past[i+down_for][j+right_for+1] + pel_past[i+down_for+1][j+right_for+1] ) // 4;                
                    R_for = round((pel_future_R.at(pel_future_r).at(pel_future_c) + pel_future_R.at(pel_future_r + 1).at(pel_future_c) + pel_future_R.at(pel_future_r).at(pel_future_c + 1) + pel_future_R.at(pel_future_r + 1).at(pel_future_c + 1)) / 4);
                    G_for = round((pel_future_G.at(pel_future_r).at(pel_future_c) + pel_future_G.at(pel_future_r + 1).at(pel_future_c) + pel_future_G.at(pel_future_r).at(pel_future_c + 1) + pel_future_G.at(pel_future_r + 1).at(pel_future_c + 1)) / 4);
                    B_for = round((pel_future_B.at(pel_future_r).at(pel_future_c) + pel_future_B.at(pel_future_r + 1).at(pel_future_c) + pel_future_B.at(pel_future_r).at(pel_future_c + 1) + pel_future_B.at(pel_future_r + 1).at(pel_future_c + 1)) / 4);
                }
            }
            if (mb_motion_backward == "1") {
                int pel_past_r = pel_r + down_bac;
                int pel_past_c = pel_c + right_bac;
                if ( ! right_half_bac && ! down_half_bac ) {
                    // pel[i][j] = pel_past[i+down_for][j+right_for];
                    R_bac = pel_past_R.at(pel_past_r).at(pel_past_c);
                    G_bac = pel_past_G.at(pel_past_r).at(pel_past_c);
                    B_bac = pel_past_B.at(pel_past_r).at(pel_past_c);
                } else if ( ! right_half_bac && down_half_bac ) {
                    // pel[i][j] = ( pel_past[i+down_for][j+right_for] + pel_past[i+down_for+1][j+right_for] ) // 2;
                    R_bac = round((pel_past_R.at(pel_past_r).at(pel_past_c) + pel_past_R.at(pel_past_r + 1).at(pel_past_c)) / 2);
                    G_bac = round((pel_past_G.at(pel_past_r).at(pel_past_c) + pel_past_G.at(pel_past_r + 1).at(pel_past_c)) / 2);
                    B_bac = round((pel_past_B.at(pel_past_r).at(pel_past_c) + pel_past_B.at(pel_past_r + 1).at(pel_past_c)) / 2);
                } else if ( right_half_bac && ! down_half_bac ) {
                    // pel[i][j] = ( pel_past[i+down_for][j+right_for] + pel_past[i+down_for][j+right_for+1] ) // 2;
                    R_bac = round((pel_past_R.at(pel_past_r).at(pel_past_c) + pel_past_R.at(pel_past_r).at(pel_past_c + 1)) / 2);
                    G_bac = round((pel_past_G.at(pel_past_r).at(pel_past_c) + pel_past_G.at(pel_past_r).at(pel_past_c + 1)) / 2);
                    B_bac = round((pel_past_B.at(pel_past_r).at(pel_past_c) + pel_past_B.at(pel_past_r).at(pel_past_c + 1)) / 2);
                } else if ( right_half_bac && down_half_bac ) {
                    // pel[i][j] = ( pel_past[i+down_for][j+right_for] + pel_past[i+down_for+1][j+right_for] + pel_past[i+down_for][j+right_for+1] + pel_past[i+down_for+1][j+right_for+1] ) // 4;
                    R_bac = round((pel_past_R.at(pel_past_r).at(pel_past_c) + pel_past_R.at(pel_past_r + 1).at(pel_past_c) + pel_past_R.at(pel_past_r).at(pel_past_c + 1) + pel_past_R.at(pel_past_r + 1).at(pel_past_c + 1)) / 4);
                    G_bac = round((pel_past_G.at(pel_past_r).at(pel_past_c) + pel_past_G.at(pel_past_r + 1).at(pel_past_c) + pel_past_G.at(pel_past_r).at(pel_past_c + 1) + pel_past_G.at(pel_past_r + 1).at(pel_past_c + 1)) / 4);
                    B_bac = round((pel_past_B.at(pel_past_r).at(pel_past_c) + pel_past_B.at(pel_past_r + 1).at(pel_past_c) + pel_past_B.at(pel_past_r).at(pel_past_c + 1) + pel_past_B.at(pel_past_r + 1).at(pel_past_c + 1)) / 4);              
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
