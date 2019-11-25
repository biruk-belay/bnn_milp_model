//#include "milp.h"
#include <iostream>
#include "milp_top.h"
#include "milp.h"

using namespace std;

int main()
{
    int i;
    float percentage_of_resource_used = 0.43;
    unsigned int lut = TOT_LUT_PYNQ * percentage_of_resource_used;
    conv_ntwk_arch conv_net[] = {{32, 3, 30, 64, 3},
                                 {30, 64, 28, 64, 3},
                                 {14, 64, 12, 128, 3},
                                 {12, 128, 10, 128, 3},
                                 {5, 128, 3, 256, 3},
                                 {3, 256, 1, 256, 3}};

    lfc_ntwk_arch lfc_net[] = {{256, 512}, {512, 512},
                               {512, 64}};

    /*********************************************************************************
        #LUT(i) = 39.76 * pe(i) + 9.69  * simd(i) + 2244 -- quadrant 1
        #LUT(i) = 95.13 * pe(i) + 19.79 * simd(i) + 1552 -- quadrant 2
        #LUT(i) = 19.79 * pe(i) + 96.13 * simd(i) + 1552  -- quadrant 3
        #LUT(i) = 121.4 * pe(i) + 90.26 * simd(i) - 1674 -- quadrant 4
    ******************************************************************************/
/*    model_variables res_model_lfc[NUM_QUADRANTS] = {{27.3, 6.39, 1570.8}, {67.2, 13.85, 1086.4},
                                                {13.853, 67.29, 1086}, {84.9, 63.18, -1171.8}};
*/
    model_variables res_model_lfc[NUM_QUADRANTS] = {{20.92, 6.39, 1233.5}, {55.2, 10.15, 900.2},
                                                {13.853, 60.9, 876}, {66.19, 54.18, -1317.8}};
/*
    model_variables res_model_conv[NUM_QUADRANTS] = {{39.76, 9.69, 2244}, {96.13, 19.79, 1552},
                                                {19.79, 96.13, 1552}, {121.4, 90.26, -1674}};
*/
    model_variables res_model_conv[NUM_QUADRANTS] = {{26.64, 20.19, 1313}, {96.67, 19.79, 645.4},
                                                {38.64, 185.8, -845.8}, {151.5, 117.8, -5177}};


#ifdef FIRST_CUT
    const unsigned int num_conv_layer   = 3;
    const unsigned int num_lfc_layer    = 0; 
    unsigned int cut_layer = 0;
    unsigned long long int num_ops_per_layer[num_conv_layer + num_lfc_layer];
    
    for(i = 0; i < num_conv_layer; i++) {
        num_ops_per_layer[i] = 2 * conv_net[i + cut_layer].IFM_CH * 
                                   conv_net[i + cut_layer].OFM_CH *  
                                   conv_net[i + cut_layer].OFM * 
                                   conv_net[i + cut_layer].OFM * 
                                   conv_net[i + cut_layer].k * 
                                   conv_net[i + cut_layer].k;

        cout << " num_ops " << i << "  " << num_ops_per_layer[i] << endl;
    }
    
    for(i = 0; i < num_lfc_layer; i++) {
        num_ops_per_layer[num_conv_layer + i] = lfc_net[i].matW * lfc_net[i].matH;
        cout << " num_ops " << num_conv_layer + i << "  " << num_ops_per_layer[num_conv_layer + i] << endl;
    }
    milp_solver(lut, num_conv_layer, num_lfc_layer, num_ops_per_layer, res_model_lfc, res_model_conv);
#endif

#ifdef SECOND_CUT
    const unsigned int num_conv_layer   = 3;
    const unsigned int num_lfc_layer    = 3;
    unsigned int cut_layer = 3; 
    unsigned long long int num_ops_per_layer[num_conv_layer + num_lfc_layer];
    
    for(i = 0; i < num_conv_layer; i++) {
        num_ops_per_layer[i] = 2 * conv_net[i + cut_layer].IFM_CH * 
                                   conv_net[i + cut_layer].OFM_CH *  
                                   conv_net[i + cut_layer].OFM * 
                                   conv_net[i + cut_layer].OFM * 
                                   conv_net[i + cut_layer].k * 
                                   conv_net[i + cut_layer].k;

        cout << " num_ops " << i << "  " << num_ops_per_layer[i] << endl;
    }
    
    for(i = 0; i < num_lfc_layer; i++) {
        num_ops_per_layer[num_conv_layer + i] = lfc_net[i].matW * lfc_net[i].matH;
        cout << " num_ops " << num_conv_layer + i << "  " << num_ops_per_layer[num_conv_layer + i] << endl;
    }
   milp_solver(lut, num_conv_layer, num_lfc_layer, num_ops_per_layer, res_model_lfc, res_model_conv);
#endif

#ifdef FULL
    const unsigned int num_conv_layer   = 6;
    const unsigned int num_lfc_layer    = 3;
    unsigned int cut_layer = 0; 
    unsigned long long int num_ops_per_layer[num_conv_layer + num_lfc_layer];
    
    for(i = 0; i < num_conv_layer; i++) {
        num_ops_per_layer[i] = 2 * conv_net[i + cut_layer].IFM_CH * 
                                   conv_net[i + cut_layer].OFM_CH *  
                                   conv_net[i + cut_layer].OFM * 
                                   conv_net[i + cut_layer].OFM * 
                                   conv_net[i + cut_layer].k * 
                                   conv_net[i + cut_layer].k;

        cout << " num_ops " << i << "  " << num_ops_per_layer[i] << endl;
    }
    
    for(i = 0; i < num_lfc_layer; i++) {
        num_ops_per_layer[num_conv_layer + i] = lfc_net[i].matW * lfc_net[i].matH;
        cout << " num_ops " << num_conv_layer + i << "  " << num_ops_per_layer[num_conv_layer + i] << endl;
    }

    milp_solver(lut, num_conv_layer, num_lfc_layer, num_ops_per_layer, res_model_lfc, res_model_conv);
#endif

    return 0;
}
