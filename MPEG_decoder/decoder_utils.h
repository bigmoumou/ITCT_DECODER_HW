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
    // vector<vector<vector<int>>> img_queue {};
    int img_q_num = 0;
    int img_r_queue [240][320];
    int img_g_queue [240][320];
    int img_b_queue [240][320];
    
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
    vector<vector<vector<int>>> pic_mb_vec;
    vector<int> mb_p_num {};
    vector<int> mp_p_i {};
    vector<vector<string>> mb_intra_vec {};
    vector<vector<vector<int>>> pic_mb_vec_p;
    vector<vector<vector<int>>> pic_mb_vec_b;
    vector<vector<int>> y_result_final {};
    vector<vector<int>> cb_result_final {};
    vector<vector<int>> cr_result_final {};
    // queue for opencv video
    deque<Mat> imageQueue;

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
    vector<vector<int>> dct_recon;

    // Fast IDCT
    vector<vector <double>> idct_table;
    // pel_past
    vector<vector<int>> pel_past_R;
    vector<vector<int>> pel_past_G;
    vector<vector<int>> pel_past_B;
    vector<vector<int>> pel_future_R;
    vector<vector<int>> pel_future_G;
    vector<vector<int>> pel_future_B;
    vector<vector<int>> pel_R;
    vector<vector<int>> pel_G;
    vector<vector<int>> pel_B;
    vector<vector<int>> pel_tmp_R;
    vector<vector<int>> pel_tmp_G;
    vector<vector<int>> pel_tmp_B;

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
    // Motion Vectors
    map<string, int> motion_vector_map = {{"00000011001", -16}, {"00000011011", -15}, {"00000011101", -14},
                                                                                {"00000011111", -13}, {"00000100001", -12}, {"00000100011", -11},
                                                                                {"0000010011", -10}, {"0000010101", -9}, {"0000010111", -8},
                                                                                {"00000111", -7}, {"00001001", -6}, {"00001011", -5},
                                                                                {"0000111", -4}, {"00011", -3}, {"0011", -2}, {"011", -1},
                                                                                {"1", 0},
                                                                                {"010", 1}, {"0010", 2}, {"00010", 3}, {"0000110", 4}, {"00001010", 5},
                                                                                {"00001000", 6}, {"00000110", 7}, {"0000010110", 8}, {"0000010100", 9},
                                                                                {"0000010010", 10}, {"00000100010", 11}, {"00000100000", 12},
                                                                                {"00000011110", 13}, {"00000011100", 14},
                                                                                {"00000011010", 15}, {"00000011000", 16}};
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
    int W1 = 2841; /* 2048*sqrt(2)*cos(1*pi/16) */
    int W2 = 2676; /* 2048*sqrt(2)*cos(2*pi/16) */
    int W3 = 2408; /* 2048*sqrt(2)*cos(3*pi/16) */
    int W5 = 1609; /* 2048*sqrt(2)*cos(5*pi/16) */
    int W6 = 1108; /* 2048*sqrt(2)*cos(6*pi/16) */
    int W7 = 565;  /* 2048*sqrt(2)*cos(7*pi/16) */
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
    void print_hex(unsigned int code);
    void print_dec(unsigned int code);
    // ---> check start & end code
    void next_start_code();
    bool is_next_start_code(int code);
    bool is_next_slice_code();
    uint32_t nextbits(int num);
    uint32_t read_bits(int num);
    // ----> other layers
    int get_cur_pos(int cur_pos, int num);
    void update_pattern_code(vector<int> & pattern_code);
    
    // Get Mapping Value
    int get_mb_address_map_s();
    string get_mb_type_map();
    int get_dct_dc_size_lum_map_s();
    int get_dct_dc_size_chr_map_s();
    int get_escape_run();
    int get_escape_level();
    int get_motion_vector_map();

    // Reconstruct I-frame
    void coded_block_pattern();
    void dct_coeff_first_s();
    void dct_coeff_next_s();
    void fill_dct_zz(int run, int level);
    void fill_dct_zz_first(int run, int level);
    // new fast idct
    void idctrow (int i);
    void idctcol (int i);
    void fast_idct();

    void reconstruct_dct(int num);
    void idct();
    void recon_pic();
    void ycbcr2rgb(bool to_buffer, bool to_output);
    void ycbcr2rgb_s(bool to_buffer, bool to_output);
    void rgb2cvmat(vector<vector<int>> cv_R, vector<vector<int>> cv_G, vector<vector<int>> cv_B);
    // Reconstruct P-frame
    void cal_motion_vector_p();
    void decode_mv();
    // Reconstruct B-frame
    void cal_motion_vector_b();

    // Overloaded Constructors
    Decoder();
};