#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <deque>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

class Decoder {
private:
    // all bits string buffer for `read_data`
    deque<uint8_t> que_buf {};
    bool zero_byte_flag = false;
    uint8_t zero_byte = 0;

    int buf_cursor = 0;
    uint8_t buf = 0;

    // Sequence
    uint32_t seq_h_code = 0;
    uint16_t h_size = 0;
    uint16_t v_size = 0;
    uint8_t p_a_r = 0;
    uint8_t p_r = 0;
    uint32_t b_r = 0;
    uint8_t m_b = 0;
    uint16_t vbv_buffer_size = 0;
    uint8_t constrained_parameter_flag = 0;
    uint8_t load_intra_quantizer_matrix = 0;
    uint8_t load_non_intra_quantizer_matrix = 0;
    uint32_t sequence_end_code = 0;
    int mb_width = 0;
    int mb_height = 0;
    vector<vector<int>> intra_quant {{8, 16, 19, 22, 26, 27, 29, 34},
                                                                     {16, 16, 22, 24, 27, 29, 34, 37},
                                                                     {19, 22, 26, 27, 29, 34, 34, 38},
                                                                     {22, 22, 26, 27, 29, 34, 37, 40},
                                                                     {22, 26, 27, 29, 32, 35, 40, 48},
                                                                     {26, 27, 29, 32, 35, 40, 48, 58},
                                                                     {26, 27, 29, 34, 38, 46, 56, 69},
                                                                     {27, 29, 35, 38, 46, 56, 69, 83}};
    vector<vector<int>> non_intra_quant {{16, 16, 16, 16, 16, 16, 16, 16},
                                                                              {16, 16, 16, 16, 16, 16, 16, 16},
                                                                              {16, 16, 16, 16, 16, 16, 16, 16},
                                                                              {16, 16, 16, 16, 16, 16, 16, 16},
                                                                              {16, 16, 16, 16, 16, 16, 16, 16},
                                                                              {16, 16, 16, 16, 16, 16, 16, 16},
                                                                              {16, 16, 16, 16, 16, 16, 16, 16},
                                                                              {16, 16, 16, 16, 16, 16, 16, 16}};
    // Group of Pictures
    uint32_t gop_start_code = 0;
    uint32_t time_code = 0;
    uint8_t closed_gop = 0;
    uint8_t broken_link = 0;
    int pic_num = 0;
    
    // Pictures
    uint32_t picture_start_code = 0;
    uint16_t temporal_reference = 0;
    uint8_t picture_coding_type = 0;  // 000 forbidden, 001 i-frame, 010 p-frame, 011 b-frame
    uint16_t vbv_delay = 0;
    uint8_t extra_bit_picture = 0;
    uint8_t extra_information_picture = 0;
    // P-frame only
    uint8_t full_pel_forward_vector = 0;
    uint8_t forward_f_code = 0;
    uint8_t forward_r_size = 0;
    uint8_t forward_f = 0;
    // B-frame only
    uint8_t full_pel_backward_vector = 0;
    uint8_t backward_f_code = 0; 
    uint8_t backward_r_size = 0;
    uint8_t backward_f = 0;    
    // Pars for calculations
    vector<vector<string>> mb_intra_vec;
    vector<vector<int>> y_result_final;
    vector<vector<int>> cb_result_final;
    vector<vector<int>> cr_result_final;
    // queue for opencv video
    deque<Mat> imageQueue;
    Mat fut_image;
    Mat image;
    Mat imageQ;
    int y_tmp[16][16];
    int cb_tmp[8][8];
    int cr_tmp[8][8];

    // Slices
    int slice_vertical_position = 0;
    int dct_dc_y_past = 0;
    int dct_dc_cb_past = 0;
    int dct_dc_cr_past = 0;
    int past_intra_address = 0;
    int mb_address = 0;
    int past_mb_address = 0;
    int mb_row = 0;
    int mb_col = 0;
    uint32_t slice_start_code = 0;
    uint8_t quantizer_scale = 0;
    uint8_t extra_bit_slice = 0;
    // forward
    int recon_right_for_prev = 0;
    int recon_down_for_prev = 0;
    int complement_horizontal_forward_r = 0;
    int complement_vertical_forward_r = 0;
    int right_for = 0;
    int down_for = 0;
    int right_half_for = 0;
    int down_half_for = 0;
    // backword
    int recon_right_bac_prev = 0;
    int recon_down_bac_prev = 0;
    int complement_horizontal_backward_r = 0;
    int complement_vertical_backward_r = 0;
    int right_bac = 0;
    int down_bac = 0;
    int right_half_bac = 0;
    int down_half_bac = 0;

    // Macroblocks
    vector<int> pattern_code;
    uint16_t mb_stuffing = 0;
    uint16_t mb_escape = 0;
    uint16_t mb_address_increment = 0;
    string mb_type = "";
    string mb_quant = "";
    string mb_motion_forward = "";
    string mb_motion_backward = "";
    string mb_pattern = "";
    string mb_intra = "";
    // P-frame only
    int motion_horizontal_forward_code = 0;
    uint8_t motion_horizontal_forward_r = 0;
    int motion_vertical_forward_code = 0;
    uint8_t motion_vertical_forward_r = 0;
    int recon_right_for = 0;
    int recon_down_for = 0;
    // B-frame only
    int motion_horizontal_backward_code = 0;
    uint8_t motion_horizontal_backward_r = 0;
    int motion_vertical_backward_code = 0;
    uint8_t motion_vertical_backward_r = 0;
    int recon_right_bac = 0;
    int recon_down_bac = 0;
    // Pars for calculations
    int cbp = 0;

    // Blocks
    int block_i = 0;
    int dct_dc_size_luminance = 0;
    uint8_t dct_dc_differential = 0;
    int dct_dc_size_chrominance = 0;
    vector<int> dct_zz;
    int dct_zz_i = 0;
    uint8_t end_of_block = 0;
    // vector<vector<int>> dct_recon;
    int dct_recon[8][8];

    // pel_past
    vector<vector<int>> pel_past_R;
    vector<vector<int>> pel_past_G;
    vector<vector<int>> pel_past_B;
    vector<vector<int>> pel_future_R;
    vector<vector<int>> pel_future_G;
    vector<vector<int>> pel_future_B;
    int pel_R [240][320];
    int pel_G [240][320];
    int pel_B [240][320];

    // Utils
    vector<vector<int>> zigzag_m {{0, 1, 5, 6, 14, 15, 27, 28},
                                                                 {2, 4, 7, 13, 16, 26, 29, 42},
                                                                 {3, 8, 12, 17, 25, 30, 41, 43},
                                                                 {9, 11, 18, 24, 31, 40, 44, 53},
                                                                 {10, 19, 23, 32, 39, 45, 52, 54},
                                                                 {20, 22, 33, 38, 46, 51, 55, 60},
                                                                 {21, 34, 37, 47, 50, 56, 59, 61},
                                                                 {35, 36, 48, 49, 57, 58, 62, 63}};
    map<string, string> mb_type_i_map = {{"1", "00001"}, {"01", "10001"}};
    map<string, string> mb_type_p_map = {{"1", "01010"}, {"01", "00010"}, {"001", "01000"}, {"00011", "00001"},
                                                                                {"00010", "11010"}, {"00001", "10010"}, {"000001", "10001"}};
    map<string, string> mb_type_b_map = {{"10", "01100"}, {"11", "01110"}, {"010", "00100"}, {"011", "00110"},
                                                                                {"0010", "01000"}, {"0011", "01010"}, {"00011", "00001"}, {"00010", "11110"},
                                                                                {"000011", "11010"}, {"000010", "10110"}, {"000001", "10001"}};
    // Macroblock Pattern
   map<string, int> mb_pattern_map = {{"111", 60}, {"1101", 4}, {"1100", 8}, {"1011", 16}, {"1010", 32},
                                                                          {"10011", 12}, {"10010", 48}, {"10001", 20}, {"10000", 40}, {"01111", 28},
                                                                          {"01110", 44}, {"01101", 52}, {"01100", 56}, {"01011", 1}, {"01010", 61},
                                                                          {"01001", 2}, {"01000", 62}, {"001111", 24}, {"001110", 36}, {"001101", 3},
                                                                          {"001100", 63}, {"0010111", 5}, {"0010110", 9}, {"0010101", 17}, {"0010100", 33},
                                                                          {"0010011", 6}, {"0010010", 10}, {"0010001", 18}, {"0010000", 34}, {"00011111", 7},
                                                                          {"00011110", 11}, {"00011101", 19}, {"00011100", 35},{"00011011",13},{"00011010",49},{"00011001",21},{"00011000",41},
                                                                          {"00010111", 14},{"00010110",50},{"00010101",22},{"00010100",42},{"00010011",15},
                                                                          {"00010010", 51},{"00010001",23},{"00010000",43},{"00001111",25},{"00001110",37},
                                                                          {"00001101", 26},{"00001100",38},{"00001011",29},{"00001010",45},{"00001001",53},
                                                                          {"00001000", 57},{"00000111",30},{"00000110",46},{"00000101",54},{"00000100",58},
                                                                          {"000000111",31},{"000000110",47},{"000000101",55},{"000000100",59},{"000000011",27},
                                                                          {"000000010",39}};
    // new fast idct
    int W1 = 2841;
    int W2 = 2676;
    int W3 = 2408;
    int W5 = 1609;
    int W6 = 1108;
    int W7 = 565;
    int16_t iclip[1024];
    int16_t *iclp;

public:
    // Main functions
    void read_data(string filename);
    
    void video_sequence();
    
    void sequence_header();
    
    void group_of_pictures();
    
    void picture();
    
    void slice();
    
    void macroblock();
    
    void block(int i);
    
    // Utils
    int sign(int num);
    void load_intra_quant();
    void load_non_intra_quant();
    void output_img();
    void next_start_code();
    bool is_next_start_code(int code);
    bool is_next_slice_code();
    uint32_t nextbits(int num);
    uint32_t read_bits(int num);
    int get_cur_pos(int cur_pos, int num);
    void update_pattern_code(vector<int> & pattern_code);

    // Get Mapping Value
    int get_mb_address_map_s();
    string get_mb_type_map();
    int get_dct_dc_size_lum_map_s();
    int get_dct_dc_size_chr_map_s();
    int get_escape_run();
    int get_escape_level();
    int get_motion_vector_map_s();
    void pel_past_2_future();

    // Reconstruct I-frame
    void coded_block_pattern();
    void dct_coeff_first_s();
    void dct_coeff_next_s3();
    void fill_dct_zz(int run, int level);
    void fill_dct_zz_first(int run, int level);
    // new fast idct
    void idctrow (int i);
    void idctcol (int i);
    void fast_idct();
    void collect_mbs();
    void recon_image_s();
    void recon_image_skip_p();
    void recon_image_skip_b();

    void reconstruct_dct(int num);
    void rgb2cvmat();
    // Reconstruct P-frame
    void cal_motion_vector_p();
    void decode_mv_s();
    // Reconstruct B-frame
    void cal_motion_vector_b();

    // Overloaded Constructors
    Decoder();
};