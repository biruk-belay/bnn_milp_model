#include <iostream>
#include <vector>
#include <cmath>
#include <climits>
#include "gurobi_c++.h"
#include "gurobi_c.h"
#include "milp.h"
#include "milp_top.h"

using namespace std;

#define BATCH_SIZE 256
#define T_reconf 4500000
typedef vector<GRBVar>          GRBVarArray;
typedef vector<GRBVarArray>     GRBVar2DArray; 
typedef vector<GRBVar2DArray>     GRBVar3DArray; 

const int t = 2;

int milp_solver(unsigned int num_lut, unsigned int num_bram, unsigned int num_conv_layer,
                unsigned int num_lfc_layer, unsigned long long int *num_ops_per_layer,
                model_variables res_model_lfc[][4], model_variables res_model_conv[][4])
{
    unsigned int MAX_LUT_PYNQ = num_lut;
    unsigned int MAX_BRAM_PYNQ = num_bram;

    unsigned int i, j, k;
    unsigned int status;

    unsigned int pe_th[t] = {16, 64}; 
    unsigned int simd_th[t] = {16, 64}; 
    unsigned long long BIG_M = 9000000000; //represents infinity
    unsigned long long BIG_M_new = 1000000000; //represents infinity
    double episilon = 0.1;
    unsigned int num_total_chunk = num_conv_layer + num_lfc_layer;
    unsigned int num_total_cuts = num_conv_layer + num_lfc_layer - 1;
    cout << "big_m is "<< BIG_M <<endl;
    try {
    
        GRBEnv env = GRBEnv();
        GRBConstr* c = NULL;
        GRBModel model = GRBModel(env);

    /**********************************************************************
         name: P
         type: integer
         func: P[i] represent the PE in each layer
     ***********************************************************************/    
    GRBVarArray P(num_conv_layer + num_lfc_layer);
    for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
        P[i] = model.addVar(2.0, PE_MAX, 0.0, GRB_INTEGER);
    }

    /**********************************************************************
        name: S
         type: integer
         func: S[i] represent the simd in each neuron on a single layer
    ***********************************************************************/

    GRBVarArray S(num_conv_layer + num_lfc_layer);
    for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
        S[i] = model.addVar(2.0, SIMD_MAX, 0.0, GRB_INTEGER);
    }

    /**************************************************************************
         name: alpha
         type: binary
         func: alpha[i] is used to choose the interval of P 
     ***************************************************************************/
    GRBVar2DArray alpha(t);
        for(i =0; i < t; i++) { 
            GRBVarArray each_alpha(num_conv_layer + num_lfc_layer);
            alpha[i] = each_alpha;
            for(k = 0; k < num_conv_layer + num_lfc_layer; k++) {
            alpha[i][k] = model.addVar(0.0,  1.0, 0.0, GRB_INTEGER);
         }
    }
    /**************************************************************************
         name: gamma
         type: binary
         func: gamma[i] is used to choose the interval of S   
     ***************************************************************************/
    GRBVar2DArray gamma(t); 
        for(i =0; i < t; i++) { 
            GRBVarArray each_alpha(num_conv_layer + num_lfc_layer);
            gamma[i] = each_alpha;
            for(k = 0; k < num_conv_layer + num_lfc_layer; k++) {
            gamma[i][k] = model.addVar(0.0,  1.0, 0.0, GRB_INTEGER);
         }
    }
     /**************************************************************************
         name: beta
         type: binary
         func: beta[i][j] is used to constrain P_i to multiples 2^x
     ***************************************************************************/
    GRBVar2DArray beta(num_conv_layer + num_lfc_layer);
        for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
            GRBVarArray each_alpha(PE_MAX);
            beta[i] = each_alpha;
            for(k = 0; k < PE_MAX; k++)
                beta[i][k] = model.addVar(0.0,  1.0, 0.0, GRB_BINARY);
         }

     /**************************************************************************
         name: delta
         type: binary
         func: delta[i][j] is used to constrain S_i to multiples of 2^x
     ***************************************************************************/
    GRBVar2DArray delta(num_conv_layer + num_lfc_layer);
        for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
            GRBVarArray each_alpha(SIMD_MAX);
            delta[i] = each_alpha;
            for(k = 0; k < SIMD_MAX; k++)
                delta[i][k] = model.addVar(0.0,  1.0, 0.0, GRB_BINARY);
         }
    /**********************************************************************
         name: lat
         type: integer
         func: lat represent the total latency of the neural net
     ***********************************************************************/
    GRBVar lat = model.addVar(1.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS);

    /**********************************************************************
         name: lut
         type: integer
         func: lut[i] represent the number of LUT in the ith layer
    ***********************************************************************/
    GRBVarArray lut(num_conv_layer + num_lfc_layer);
    for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
        lut[i] = model.addVar(1.0, MAX_LUT_PYNQ, 0.0, GRB_CONTINUOUS);
    }

    /**********************************************************************
         name: bram
         type: integer
         func: bram[i] represent the number of bram in the ith layer
    ***********************************************************************/
    GRBVarArray bram(num_conv_layer + num_lfc_layer);
    for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
        bram[i] = model.addVar(1.0, MAX_LUT_PYNQ, 0.0, GRB_CONTINUOUS);
    }
 
   /**********************************************************************
         name: tau
         type: integer
         func: 
    ***********************************************************************/
    GRBVarArray tau(num_conv_layer + num_lfc_layer);
    for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
        tau[i] = model.addVar(0.0, 65536, 0.0, GRB_INTEGER);
    }

    /**************************************************************************
         name: lamda
         type: binary
         func: lambda[i][0] is used to constrain tau_i to multiples of 2^x 
     ***************************************************************************/
        GRBVar2DArray lambda(num_conv_layer + num_lfc_layer);
#if defined(FIRST_CUT) || defined(FULL)
        GRBVarArray beta_first(SIMD_EXP * PE_EXP);
        lambda[0] = beta_first;
        for(k = 0; k < SIMD_EXP * PE_EXP; k++)
            lambda[0][k] = model.addVar(0.0,  1.0, 0.0, GRB_BINARY);
        
        for(i = 1; i < num_conv_layer + num_lfc_layer; i++) {
            GRBVarArray each_alpha(SIMD_EXP + PE_EXP);
            lambda[i] = each_alpha;
            for(k = 0; k < (SIMD_EXP + PE_EXP); k++)
                lambda[i][k] = model.addVar(0.0,  1.0, 0.0, GRB_BINARY); 
        }
#else
        for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
            GRBVarArray each_alpha(SIMD_EXP + PE_EXP);
            lambda[i] = each_alpha;
            for(k = 0; k < (SIMD_EXP + PE_EXP); k++)
                lambda[i][k] = model.addVar(0.0,  1.0, 0.0, GRB_BINARY); 
        }
#endif
    /**************************************************************************
         name: layer_lat
         type: binary
         func: layer_lat[i] 
     ***************************************************************************/
    GRBVarArray layer_lat(num_conv_layer + num_lfc_layer);
    for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
        layer_lat[i] = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
    }

    /**************************************************************************
         name: max_lat
         type: binary
         func: max_lat is the maximum of all the latencies of each layer 
     ***************************************************************************/
       GRBVar max_lat;
        max_lat = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS);

    /**************************************************************************
         name: x
         type: binary
         func: x[i] is used to signify the cuts   
     ***************************************************************************/
    GRBVarArray x (num_conv_layer + num_lfc_layer - 1);
        for(i = 0; i < num_conv_layer + num_lfc_layer - 1; i++) {
            x[i] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
         }

     /**************************************************************************
         name: y
         type: binary
         func: y[i][j] = 1 if the i-th layer is in the j-th chunk
     ***************************************************************************/
    GRBVar2DArray y(num_conv_layer + num_lfc_layer);
        for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
            GRBVarArray each_alpha(num_conv_layer + num_lfc_layer);
            y[i] = each_alpha;
            for(k = 0; k < num_conv_layer + num_lfc_layer; k++)
                y[i][k] = model.addVar(0.0,  1.0, 0.0, GRB_BINARY);
         }
    /**************************************************************************
         name: phi_max
         type: real
         func: phi_max[j] is the maximum latency in the j-th chunk   
     ***************************************************************************/
    GRBVarArray phi_max(num_conv_layer + num_lfc_layer);
        for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
            phi_max[i] = model.addVar(0.0,  GRB_INFINITY, 0.0, GRB_CONTINUOUS);
         }
    /**************************************************************************
         name: phi_tot
         type: real
         func: phi_tot[j] is the maximum latency in the j-th chunk   
     ***************************************************************************/
    GRBVarArray phi_tot(num_conv_layer + num_lfc_layer);
        for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
            phi_tot[i] = model.addVar(0.0,  GRB_INFINITY, 0.0, GRB_CONTINUOUS);
         }

     /**************************************************************************
         name: a
         type: real
         func: a[i][j] 
     ***************************************************************************/
    GRBVar3DArray a(t);
        for(j = 0; j < t; j++) {
            GRBVar2DArray each_alpha(num_conv_layer + num_lfc_layer);
            a[j] = each_alpha;
            for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
                GRBVarArray each_alpha_new(num_conv_layer + num_lfc_layer);
                a[j][i] = each_alpha_new;
                for(k = 0; k < num_conv_layer + num_lfc_layer; k++)
                    a[j][i][k] = model.addVar(0.0,  GRB_INFINITY, 0.0, GRB_CONTINUOUS);
            }
        } 
    /**************************************************************************
         name: b
         type: real
         func: b[i][j] = 1 if the i-th layer is in the j-th chunk
    ***************************************************************************/
    GRBVar2DArray b(num_conv_layer + num_lfc_layer);
        for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
            GRBVarArray each_alpha(num_conv_layer + num_lfc_layer);
            b[i] = each_alpha;
            for(k = 0; k < num_conv_layer + num_lfc_layer; k++)
                b[i][k] = model.addVar(0.0,  GRB_INFINITY, 0.0, GRB_CONTINUOUS);
         }
    
    /**************************************************************************
         name: d
         type: real
         func: d[i][j] = 1 if the i-th layer is in the j-th chunk
     ***************************************************************************/
    GRBVar2DArray d(num_conv_layer + num_lfc_layer);
        for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
            GRBVarArray each_alpha(num_conv_layer + num_lfc_layer);
            d[i] = each_alpha;
            for(k = 0; k < num_conv_layer + num_lfc_layer; k++)
                d[i][k] = model.addVar(0.0,  GRB_INFINITY, 0.0, GRB_CONTINUOUS);
         }

    /**************************************************************************
         name: R
         type: real
         func: R[i][j] is the i type of resource in the j-th chunk
     ***************************************************************************/
    GRBVar2DArray R(3);
        for(i = 0; i < 3; i++) {
            GRBVarArray each_alpha(num_conv_layer + num_lfc_layer);
            R[i] = each_alpha;
            for(k = 0; k < num_conv_layer + num_lfc_layer; k++)
                R[i][k] = model.addVar(0.0,  MAX_LUT_PYNQ, 0.0, GRB_CONTINUOUS);
         }

    model.update();

    /****************************************************************************
    Constr 0.1:
 
     ****************************************************************************/
    for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
        GRBLinExpr exp;
        for(j = 0; j < num_total_chunk; j++) {
            exp += y[i][j];    
        }
        model.addConstr(exp == 1, "900");
    }
    
    /****************************************************************************
    Constr 0.2:
 
    ****************************************************************************/
     for(i = 0; i < num_total_cuts; i++) {
        for(j = 0; j < num_total_chunk; j++) {
            model.addConstr(y[i][j] <= y[i+1][j] + x[i] * BIG_M, "901");
            model.addConstr(y[i+1][j] <= y[i][j] + x[i] * BIG_M, "902");
        }
     }

    /****************************************************************************
    Constr 0.3:

     ****************************************************************************/
     for(i = 0; i < num_total_cuts; i++) {
        GRBLinExpr exp, exp_x;
        for(j = 0; j < num_total_chunk; j++) {
            model.addConstr(y[i][j] + y[i+1][j] <= 1 + (1 - x[i]) * BIG_M, "903");
        }
     }

    /****************************************************************************
    Constr 0.4:

    ****************************************************************************/
     GRBLinExpr exp_x;
     for(i = 0; i < num_total_cuts; i++) {
            exp_x += x[i]; 
     }
     model.addConstr(exp_x <= 3, "904");
     //model.addConstr(exp_x >= 1);
     //model.addConstr(x[2] == 1);
    /****************************************************************************
    Constr 1.1: alpha[][0] == 0 iff pe < pe_th else alpha[][0] == 1
                alpha[][1] == 0 iff simd < pe_th else alpha[][1] == 1

                alpha[][0] and alpha[][1] are used to decide which model to use
                to calculate the resource consumption
    ****************************************************************************/ 
     for(k = 0; k < t; k++) {
         for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
            model.addConstr(P[i] - episilon >= pe_th[k] - BIG_M_new * (1 - alpha[k][i]), "1");
            model.addConstr(P[i] - episilon <= pe_th[k] + BIG_M_new * alpha[k][i], "2");
         }
     }
     
     for(k = 0; k < t; k++) {
         for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
             model.addConstr(S[i] - episilon >= simd_th[k] - BIG_M_new * (1 - gamma[k][i]), "3");
             model.addConstr(S[i] - episilon <= simd_th[k] + BIG_M_new * gamma[k][i], "4");
         }
     }

    /******************************************************************
        Constr 1.2.0: lut[i] is the amount of LUTs in the i-th layer. 
                    This constraint enforces the amount of LUT in each 
                    layer                
    ******************************************************************/
    GRBLinExpr exp;
    for(i = 0; i < num_conv_layer; i++) {
        //LUT section
        model.addConstr(lut[i] >= res_model_conv[0][0].pe_coeff   * P[i]   +
                                res_model_conv[0][0].simd_coeff * S[i] +
                                res_model_conv[0][0].intercept -
                                ((alpha[0][i] + gamma[0][i]) * BIG_M_new), "5");

        model.addConstr(lut[i] >= res_model_conv[0][1].pe_coeff   * P[i]   +
                                res_model_conv[0][1].simd_coeff * S[i] +
                                res_model_conv[0][1].intercept -
                                ((1 - alpha[0][i] + gamma[0][i]) * BIG_M_new), "6");

        model.addConstr(lut[i] >= res_model_conv[0][2].pe_coeff   * P[i]   +
                                res_model_conv[0][2].simd_coeff * S[i] +
                                res_model_conv[0][2].intercept -
                                ((1 + alpha[0][i] - gamma[0][i]) * BIG_M_new), "7");

         model.addConstr(lut[i] >= res_model_conv[0][3].pe_coeff   * P[i]   +
                                res_model_conv[0][3].simd_coeff * S[i] +
                                res_model_conv[0][3].intercept -
                                ((2 - alpha[0][i] - gamma[0][i]) * BIG_M_new), "8");

        //Bram section
        model.addConstr(bram[i] >= res_model_conv[1][0].pe_coeff   * P[i]   +
                                res_model_conv[1][0].simd_coeff * S[i] +
                                res_model_conv[1][0].intercept -
                                ((alpha[1][i] + gamma[1][i]) * BIG_M_new), "5_1");

        model.addConstr(bram[i] >= res_model_conv[1][1].pe_coeff   * P[i]   +
                                res_model_conv[1][1].simd_coeff * S[i] +
                                res_model_conv[1][1].intercept -
                                ((1 - alpha[1][i] + gamma[1][i]) * BIG_M_new), "6_1");

        model.addConstr(bram[i] >= res_model_conv[1][2].pe_coeff   * P[i]   +
                                res_model_conv[1][2].simd_coeff * S[i] +
                                res_model_conv[1][2].intercept -
                                ((1 + alpha[1][i] - gamma[1][i]) * BIG_M_new), "7_1");

         model.addConstr(bram[i] >= res_model_conv[1][3].pe_coeff   * P[i]   +
                                res_model_conv[1][3].simd_coeff * S[i] +
                                res_model_conv[1][3].intercept -
                                ((2 - alpha[1][i] - gamma[1][i]) * BIG_M_new), "8_1");
   
    }
    
    for(i = num_conv_layer; i < num_lfc_layer + num_conv_layer; i++) {
        //LUT section
        model.addConstr(lut[i] >= res_model_lfc[0][0].pe_coeff   * P[i]   +
                                res_model_lfc[0][0].simd_coeff * S[i] + 
                                res_model_lfc[0][0].intercept -
                                (alpha[0][i] + gamma[0][i]) * BIG_M_new, "4");
        
        model.addConstr(lut[i] >= res_model_lfc[0][1].pe_coeff   * P[i]   +
                                res_model_lfc[0][1].simd_coeff * S[i] + 
                                res_model_lfc[0][1].intercept -
                                (1 - alpha[0][i] + gamma[0][i]) * BIG_M_new, "5");
        
        model.addConstr(lut[i] >= res_model_lfc[0][2].pe_coeff   * P[i]   +
                                res_model_lfc[0][2].simd_coeff * S[i] +
                                res_model_lfc[0][2].intercept -
                                (1 + alpha[0][i] - gamma[0][i]) * BIG_M_new, "6");

        model.addConstr(lut[i] >= res_model_lfc[0][3].pe_coeff   * P[i]   +
                                res_model_lfc[0][3].simd_coeff * S[i] +
                                res_model_lfc[0][3].intercept -
                                (2 - alpha[0][i] - gamma[0][i]) * BIG_M_new, "7"); 


        //bram section
        model.addConstr(bram[i] >= res_model_lfc[1][0].pe_coeff   * P[i]   +
                                res_model_lfc[1][0].simd_coeff * S[i] +
                                res_model_lfc[1][0].intercept -
                                (alpha[1][i] + gamma[1][i]) * BIG_M_new, "4_1");

        model.addConstr(bram[i] >= res_model_lfc[1][1].pe_coeff   * P[i]   +   
                                res_model_lfc[1][1].simd_coeff * S[i] +      
                                res_model_lfc[1][1].intercept -
                                (1 - alpha[1][i] + gamma[1][i]) * BIG_M_new, "5_1");

        model.addConstr(bram[i] >= res_model_lfc[1][2].pe_coeff   * P[i]   +   
                                res_model_lfc[1][2].simd_coeff * S[i] +
                                res_model_lfc[1][2].intercept -
                                (1 + alpha[1][i] - gamma[1][i]) * BIG_M_new, "6_1");

        model.addConstr(bram[i] >= res_model_lfc[1][3].pe_coeff   * P[i]   +
                                res_model_lfc[1][3].simd_coeff * S[i] +
                                res_model_lfc[1][3].intercept -
                                (2 - alpha[1][i] - gamma[1][i]) * BIG_M_new, "7_1");
          
    }
   
    /******************************************************************
        Constr 1.2.1: This is a mirror of constr 1.2.0 to enforce
                    equality
    ******************************************************************/
    for(i = 0; i < num_conv_layer; i++) {    
        // LUT section
        model.addConstr((res_model_conv[0][0].pe_coeff  *  P[i]   +
                            res_model_conv[0][0].simd_coeff *  S[i] + 
                            res_model_conv[0][0].intercept)  >= (lut[i]  - 
                            ((alpha[0][i] + gamma[0][i]) * BIG_M_new)), "9");
        
        model.addConstr((res_model_conv[0][1].pe_coeff  * P[i]    +
                            res_model_conv[0][1].simd_coeff * S[i]  + 
                            res_model_conv[0][1].intercept) >= (lut[i]   -
                            (1 - alpha[0][i] + gamma[0][i]) * BIG_M_new), "10");
        
        model.addConstr((res_model_conv[0][2].pe_coeff  * P[i]    +
                            res_model_conv[0][2].simd_coeff * S[i]  +
                            res_model_conv[0][2].intercept)  >= (lut[i]  -      
                            ((1 + alpha[0][i] - gamma[0][i]) * BIG_M_new)), "11");

        model.addConstr((res_model_conv[0][3].pe_coeff  * P[i]   +
                            res_model_conv[0][3].simd_coeff * S[i] +
                            res_model_conv[0][3].intercept) >= (lut[i]  -
                            ((2 - alpha[0][i] - gamma[0][i]) * BIG_M_new)), "12");

        //bram section
        model.addConstr((res_model_conv[1][0].pe_coeff  *  P[i]   +
                            res_model_conv[1][0].simd_coeff *  S[i] +
                            res_model_conv[1][0].intercept)  >= (bram[i]  -
                            ((alpha[1][i] + gamma[1][i]) * BIG_M_new)), "9");

        model.addConstr((res_model_conv[1][1].pe_coeff  * P[i]    +
                            res_model_conv[1][1].simd_coeff * S[i]  +
                            res_model_conv[1][1].intercept) >= (bram[i]   -
                            (1 - alpha[1][i] + gamma[1][i]) * BIG_M_new), "10");

        model.addConstr((res_model_conv[1][2].pe_coeff  * P[i]    +
                            res_model_conv[1][2].simd_coeff * S[i]  +
                            res_model_conv[1][2].intercept)  >= (bram[i]  -
                            ((1 + alpha[1][i] - gamma[1][i]) * BIG_M_new)), "11");

        model.addConstr((res_model_conv[1][3].pe_coeff  * P[i]   +
                            res_model_conv[1][3].simd_coeff * S[i] +
                            res_model_conv[1][3].intercept) >= (bram[i]  -
                            ((2 - alpha[1][i] - gamma[1][i]) * BIG_M_new)), "12");

    }   


    for(i = num_conv_layer; i < num_lfc_layer + num_conv_layer; i++) {
        model.addConstr(res_model_lfc[0][0].pe_coeff   *  P[i]   +
                            res_model_lfc[0][0].simd_coeff *  S[i] + 
                            res_model_lfc[0][0].intercept  >= lut[i]  - 
                            (alpha[0][i] + gamma[0][i]) * BIG_M_new, "9");
        
        model.addConstr(res_model_lfc[0][1].pe_coeff   * P[i]    +
                            res_model_lfc[0][1].simd_coeff * S[i]  + 
                            res_model_lfc[0][1].intercept >= lut[i]   -
                            (1 - alpha[0][i] + gamma[0][i]) * BIG_M_new, "10");
        
        model.addConstr(res_model_lfc[0][2].pe_coeff   * P[i]    +
                            res_model_lfc[0][2].simd_coeff * S[i]  +
                            res_model_lfc[0][2].intercept  >= lut[i]  - 
                            (1 + alpha[0][i] - gamma[0][i]) * BIG_M_new, "11");

        model.addConstr(res_model_lfc[0][3].pe_coeff   * P[i]   +
                            res_model_lfc[0][3].simd_coeff * S[i] +
                            res_model_lfc[0][3].intercept >= lut[i]  -
                            (2 - alpha[0][i] - gamma[0][i]) * BIG_M_new, "12");
        

        //bram section
        model.addConstr(res_model_lfc[1][0].pe_coeff   *  P[i]   +
                            res_model_lfc[1][0].simd_coeff *  S[i] +
                            res_model_lfc[1][0].intercept  >= bram[i]  -
                            (alpha[1][i] + gamma[1][i]) * BIG_M_new, "9");

        model.addConstr(res_model_lfc[1][1].pe_coeff   * P[i]    +
                            res_model_lfc[1][1].simd_coeff * S[i]  +
                            res_model_lfc[1][1].intercept >= bram[i]   -
                            (1 - alpha[1][i] + gamma[1][i]) * BIG_M_new, "10");

        model.addConstr(res_model_lfc[1][2].pe_coeff   * P[i]    +
                            res_model_lfc[1][2].simd_coeff * S[i]  +
                            res_model_lfc[1][2].intercept  >= bram[i]  -
                            (1 + alpha[1][i] - gamma[1][i]) * BIG_M_new, "11");

        model.addConstr(res_model_lfc[1][3].pe_coeff   * P[i]   +
                            res_model_lfc[1][3].simd_coeff * S[i] +
                            res_model_lfc[1][3].intercept >= bram[i]  -
                            (2 - alpha[1][i] - gamma[1][i]) * BIG_M_new, "12");
  
    }

   /**********************************************************************
        Constr 1.3: Constraints related to chunks and resources
   **********************************************************************/    
    for(j = 0; j < num_total_chunk; j++) {    
        GRBLinExpr exp_lut, exp_bram;    
        for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
            model.addConstr(a[0][i][j] <= y[i][j] * BIG_M_new, "915");
            model.addConstr(a[0][i][j] <= lut[i], "916");
            model.addConstr(a[0][i][j] >= lut[i] - (1 - y[i][j]) * BIG_M_new, "917");
        
            exp_lut += a[0][i][j];
          
            model.addConstr(a[1][i][j] <= y[i][j] * BIG_M_new, "915_1");
            model.addConstr(a[1][i][j] <= bram[i], "916_1");
            model.addConstr(a[1][i][j] >= bram[i] - (1 - y[i][j]) * BIG_M_new, "917_1");

            exp_bram += a[1][i][j];
            
        }
            model.addConstr(exp_lut  <=  MAX_LUT_PYNQ, "919");
            model.addConstr(exp_bram <=  MAX_BRAM_PYNQ, "919_1");
            model.addConstr(R[0][j]  ==  exp_lut, "918");
            model.addConstr(R[1][j]  ==  exp_bram, "918_1");    
    }

   /**********************************************************************
        Constr 1.4: The number of PE must be greater than SIMD in a layer
                    Part of a FINN constraint
    **********************************************************************/
    for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
//        model.addConstr(pe[i] >= simd[i], "13");
//        model.addConstr(pe[i] >= 64, "14");
    }

   /**********************************************************************
        Constr 1.5: for each layer i
                        for j = 0.... log(PE_MAX)
                            pe[i] = sum (2^i * beta_pe[i][j]
                    
                    for each layer i
                        for j = 0.... log(PE_MAX)
                            sum beta_pe[i][j] = 1
    **********************************************************************/
    for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
        GRBLinExpr exp1, exp2;
        for(j = 0; j < PE_EXP; j++) {
            exp2 += pow(2, j+1) * beta[i][j];
            exp1 += beta[i][j];
        }
            model.addConstr(P[i] == exp2, "15");
            model.addConstr(exp1 == 1, "16");
    }

   /**********************************************************************
        Constr 1.6: similar to constraint 1.5 but for simd
    **********************************************************************/
#if defined(FIRST_CUT) || defined(FULL)        
    GRBLinExpr exp5, exp6;
    for(j = 0; j < SIMD_EXP; j++) {    
        //IF the S[0] issue is not resolved this constraint should be enabled
        //exp6 += 3 * 1 * delta[0][j];
        //else enable this constraint
        exp6 += 3 * (j+1) * delta[0][j];
        exp5 += delta[0][j];
    }       
 
    //IF the S[0] issue is not resolved this constraint should be enabled
    //model.addConstr(S[0] == 3, "125_1");
    //else enable this constraint
    model.addConstr(S[0] == exp6, "125");     
    
    model.addConstr(exp5 == 1, "126");
    for(i = 1; i < num_conv_layer + num_lfc_layer; i++) {
        GRBLinExpr exp5, exp6;
        for(j = 0; j < SIMD_EXP; j++) {
            exp6 += pow(2, j+1) * delta[i][j];
            exp5 += delta[i][j];
        }
            model.addConstr(S[i] == exp6, "127");
            model.addConstr(exp5 == 1, "128");
    }
#else
    for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
        GRBLinExpr exp5, exp6;
        for(j = 0; j < SIMD_EXP; j++) {
            exp6 += pow(2, j+1) * delta[i][j];
            exp5 += delta[i][j];
        }
            model.addConstr(S[i] == exp6, "155");
            model.addConstr(exp5 == 1, "166");
    }
#endif

   /**********************************************************************
        Constr 1.7: similar to constr 1.5 but for tau
    **********************************************************************/
#if defined(FIRST_CUT) || defined(FULL)
    GRBLinExpr exp7, exp8;
    for(i = 0; i < SIMD_EXP; i++) { 
        for(j = 0; j < PE_EXP; j++) {
            //IF the S[0] issue is not resolved this constraint should be enabled
            //exp8 += 3 * 1 * pow(2, j+1) * lambda[0][i * SIMD_EXP + j];
            //else enable this constraint
            exp8 += 3 * (i+1) * pow(2, j+1) * lambda[0][i * SIMD_EXP + j];
            exp7 += lambda[0][i * SIMD_EXP + j];    
        }
    }
    model.addConstr(tau[0] == exp8, "156");    
    model.addConstr(exp7 == 1, "165");

    for(i = 1; i < num_conv_layer + num_lfc_layer; i++) {
        GRBLinExpr exp5, exp6;
        for(j = 0; j < SIMD_EXP + PE_EXP; j++) {
            exp6 += pow(2, j+1) * lambda[i][j];
            exp5 += lambda[i][j];
        }
            model.addConstr(tau[i] == exp6, "156");
            model.addConstr(exp5 == 1, "165");
    }
#else
    GRBLinExpr exp5, exp6;
    for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
        GRBLinExpr exp5, exp6;
        for(j = 0; j < SIMD_EXP + PE_EXP; j++) {
            exp6 += pow(2, j+1) * lambda[i][j];
            exp5 += lambda[i][j];
        }
            model.addConstr(tau[i] == exp6, "156");
            model.addConstr(exp5 == 1, "165");
    }
#endif
    /**********************************************************************
        Constr 1.8: latency constraint
    **********************************************************************/
#if defined(FIRST_CUT) || defined(FULL)
    for(j = 0; j < SIMD_EXP; j++) {
            //IF the S[0] issue is not resolved this constraint should be enabled
            //model.addConstr(tau[0] >= 3 * 1 * P[0] - (1 - delta[0][j]) * BIG_M, "171");
            //model.addConstr(3 * 1 * P[0] >= tau[0] - (1 - delta[0][j]) * BIG_M, "181");
            //else enable this constraint
            model.addConstr(tau[0] >= 3 * (j+1) * P[0] - (1 - delta[0][j]) * BIG_M, "171"); 
            model.addConstr(3 * (j+1) * P[0] >= tau[0] - (1 - delta[0][j]) * BIG_M, "181");
        }
    for(i = 1; i < num_conv_layer + num_lfc_layer; i++) {
        for(j = 0; j < SIMD_EXP; j++) {
            model.addConstr(tau[i] >= pow(2, j+1) * P[i] - (1 - delta[i][j]) * BIG_M, "172");
            model.addConstr(pow(2, j+1) * P[i] >= tau[i] - (1 - delta[i][j]) * BIG_M, "182");
        }
       //exp3 += tau[i];
    }

#else
    for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
        for(j = 0; j < SIMD_EXP; j++) {
            model.addConstr(tau[i] >= pow(2, j+1) * P[i] - (1 - delta[i][j]) * BIG_M, "173");
            model.addConstr(pow(2, j+1) * P[i] >= tau[i] - (1 - delta[i][j]) * BIG_M, "184");
        }
       //exp3 += tau[i];
    }
#endif


#if defined(FIRST_CUT) || defined(FULL)
    for(i=0; i < SIMD_EXP; i++){
       for(j = 0; j < PE_EXP; j++) {
         //IF the S[0] issue is not resolved this constraint should be enabled
         //model.addConstr((num_ops_per_layer[0] / (3 * 1 * pow(2, j+1))) >=  layer_lat[0] - (1 - lambda[0][i * SIMD_EXP + j]) * BIG_M, "19");
         //model.addConstr(layer_lat[0]  >= (num_ops_per_layer[0] / (3 * 1 * pow(2, j+1))) - (1 - lambda[0][i * SIMD_EXP + j]) * BIG_M, "20");
         //else enable this constraint
         model.addConstr((num_ops_per_layer[0] / (3 * (i+1) * pow(2, j+1))) >=  layer_lat[0] - (1 - lambda[0][i * SIMD_EXP + j]) * BIG_M, "19");
         model.addConstr(layer_lat[0]  >= (num_ops_per_layer[0] / (3 * (i + 1) * pow(2, j+1))) - (1 - lambda[0][i * SIMD_EXP + j]) * BIG_M, "20");
        }
    }
    
    for(i = 1; i < num_conv_layer; i++) {
       for(j = 0; j < SIMD_EXP + PE_EXP; j++) {
          model.addConstr((num_ops_per_layer[i] / pow(2, j+1)) >=  layer_lat[i] - (1 - lambda[i][j]) * BIG_M, "19");
          model.addConstr(layer_lat[i]  >= (num_ops_per_layer[i] / pow(2, j+1)) - (1 - lambda[i][j]) * BIG_M, "20");
       }
    }

    for(i = num_conv_layer; i < num_conv_layer + num_lfc_layer; i++) {
        for(j = 0; j < SIMD_EXP + PE_EXP; j++) {
            model.addConstr((num_ops_per_layer[i] / pow(2, j+1)) >=  layer_lat[i] - (1 - lambda[i][j]) * BIG_M, "21");
            model.addConstr(layer_lat[i] >= (num_ops_per_layer[i] / pow(2, j+1))- (1 - lambda[i][j]) * BIG_M, "22");
        }
    }
#else
    for(i = 0; i < num_conv_layer; i++) {
        for(j = 0; j < SIMD_EXP + PE_EXP; j++) {
            model.addConstr((num_ops_per_layer[i] / pow(2, j+1)) >=  layer_lat[i] - (1 - lambda[i][j]) * BIG_M, "19");
            model.addConstr(layer_lat[i]  >= (num_ops_per_layer[i] / pow(2, j+1)) - (1 - lambda[i][j]) * BIG_M, "20");
        }
    }

    for(i = num_conv_layer; i < num_conv_layer + num_lfc_layer; i++) {
        for(j = 0; j < SIMD_EXP + PE_EXP; j++) {
            model.addConstr((num_ops_per_layer[i] / pow(2, j+1)) >=  layer_lat[i] - (1 - lambda[i][j]) * BIG_M, "21");
            model.addConstr(layer_lat[i] >= (num_ops_per_layer[i] / pow(2, j+1))- (1 - lambda[i][j]) * BIG_M, "22");
        }
    }
#endif

    /**********************************************************************
        Constr 1.9: new latency constraints related to delay chunks
    **********************************************************************/
    for(j = 0; j < num_total_chunk; j++) {
        GRBLinExpr exp2;
        for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
            model.addConstr(b[i][j] <= y[i][j] * BIG_M, "905");
            model.addConstr(b[i][j] <= layer_lat[i], "906");
            model.addConstr(b[i][j] >= layer_lat[i] - (1 - y[i][j]) * BIG_M, "907");

            exp2 += b[i][j];
        }
        model.addConstr(phi_tot[j] == exp2, "908");
    }
    
    for(j = 0; j < num_total_chunk; j++) {
        GRBLinExpr exp;
        for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
            model.addConstr(d[i][j] <= y[i][j] * BIG_M, "909");
            model.addConstr(d[i][j] <= layer_lat[i], "910");
            model.addConstr(d[i][j] >= layer_lat[i] - (1 - y[i][j]) * BIG_M, "911");
        
            model.addConstr(phi_max[j] >= d[i][j], "912");
        }
    }   
    
    GRBLinExpr exp_lat;
/*    for(i = 0;  i < num_conv_layer + num_lfc_layer; i++)
        exp4 += layer_lat[i];
    */ 
/*
    //Updated delay modeling   
    GRBLinExpr exp4, exp3; 
    for(i = 0;  i < num_conv_layer + num_lfc_layer; i++)
        model.addConstr(max_lat >= layer_lat[i], "121");
        
    for(i = 0;  i < num_conv_layer + num_lfc_layer; i++)
        exp3 += layer_lat[i];
            
    exp4 = (BATCH_SIZE - 1) * max_lat + exp3;
  */

    for(j = 0; j < num_total_chunk; j++) {
        exp_lat += ((BATCH_SIZE - 1) * phi_max[j] + phi_tot[j]);
        //exp_lat += phi_max[j] + phi_tot[j];
        //exp_lat += phi_max[j];
        //model.addConstr(max_lat >= phi_max[j]);
    }

    //exp_lat = max_lat;
    GRBLinExpr exp_reconf, exp_obj;
    for(i = 0; i < num_total_cuts; i++)
        exp_reconf += T_reconf * x[i];
       
    //exp_reconf = 0; 
    exp_obj = exp_reconf + exp_lat;
    /**********************************************************************
        Objective function: minimize the latency
    **********************************************************************/
    model.setObjective(exp_obj,  GRB_MINIMIZE);
    model.set(GRB_IntParam_NumericFocus, 2);
    model.set(GRB_DoubleParam_IntFeasTol, 1e-9);
    //start optimization
    model.optimize();
    
    status = model.get(GRB_IntAttr_Status);
    
    if(status == GRB_OPTIMAL) {

    /**********************************************************************
                print outputs
    **********************************************************************/

//  PE and SIMD allocation section
    int total_luts_used = 0;
    
    for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {    
//        total_luts_used += lut[i].get(GRB_DoubleAttr_X);    
        cout << " pe " << P[i].get(GRB_DoubleAttr_X) << "\tsimd " 
            << S[i].get(GRB_DoubleAttr_X) << "\t alpha " 
            << alpha[0][i].get(GRB_DoubleAttr_X) << "\t" 
            << gamma[0][i].get(GRB_DoubleAttr_X) << "\t lut " 
            << lut[i].get(GRB_DoubleAttr_X) << "\t alpha_bram \t" 
            << alpha[1][i].get(GRB_DoubleAttr_X) << "\t " 
            << gamma[1][i].get(GRB_DoubleAttr_X) << "\t" 
            << bram[i].get(GRB_DoubleAttr_X) << endl;    
    }

    cout<< endl;

    for(j = 0; j < num_conv_layer + num_lfc_layer; j++) {
        cout << "luts in chunk "<< j << "  " << R[0][j].get(GRB_DoubleAttr_X) <<endl;
    }
    
    cout <<endl;
    for(j = 0; j < num_conv_layer + num_lfc_layer; j++) {
        cout << "bram in chunk " << j << "  " << R[1][j].get(GRB_DoubleAttr_X) <<endl;
    }

    //    cout << " total luts used " << total_luts_used << endl;


    cout<< "beta" <<endl;
    for(i = 0; i < num_conv_layer; i++){
        cout << "layer " << i << "\t";
        for(j = 0; j < PE_EXP; j++)
            cout << "\t" << beta[i][j].get(GRB_DoubleAttr_X);
        cout <<endl;
    }
    
    cout<< "delta" <<endl;
    int latency = 0;
    for(i = 0; i < num_conv_layer; i++){
        latency += tau[i].get(GRB_DoubleAttr_X);
        cout << "layer " << i << "\t";
        for(j = 0; j < SIMD_EXP; j++)
            cout << "\t" << delta[i][j].get(GRB_DoubleAttr_X);
        cout <<endl;
    }
    
    cout << endl;

    for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
        cout<< "PE * SIMD \t" << tau[i].get(GRB_DoubleAttr_X);
        for(j = 0; j < SIMD_EXP + PE_EXP; j++)
            cout << "\t" << lambda[i][j].get(GRB_DoubleAttr_X);
        cout <<endl;
    }
    cout <<endl;

    //Prints related to chunks and allocations of layers in chunks
    int num_chunks = 0, num_cuts = 0;
    for(i = 0; i < num_conv_layer + num_lfc_layer - 1; i++) {
        num_chunks += x[i].get(GRB_DoubleAttr_X); 
    }

    cout <<"The cuts are between "<<endl;
    cout <<"        ";
    for(j = 0; j < num_conv_layer + num_lfc_layer - 1; j++)
       cout << "\tx["<<j<<"]";
    cout <<endl;
    cout <<"        ";
    for(i = 0; i < num_conv_layer + num_lfc_layer - 1; i++)
        if(x[i].get(GRB_DoubleAttr_X) == 1)
            cout<<"\t 1";
        else
            cout <<"\t 0";
    cout <<endl <<endl;

    cout << "total number of chunks = " << num_chunks + 1 <<endl;
    
    cout <<"        ";
    for(j = 0; j < num_conv_layer + num_lfc_layer; j++)
       cout << "\tch["<<j<<"]"; 
    cout <<endl;

    for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
        cout << "layer " << i << "\t";
        for(j = 0; j < num_conv_layer + num_lfc_layer; j++) {
            if(y[i][j].get(GRB_DoubleAttr_X) == 1.0)
                cout << "\t *";
            else
                cout <<"\tx";
        }
        
        cout <<endl;
    }
    
    cout << "y[i][j] "<<endl;
    for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
        cout << "layer " << i << "\t";
        for(j = 0; j < num_conv_layer + num_lfc_layer; j++) {
            cout << "\t" << y[i][j].get(GRB_DoubleAttr_X);
        }

        cout <<endl;
    }

    cout <<"        ";
    for(j = 0; j < num_conv_layer + num_lfc_layer; j++)
       cout << "\tch["<<j<<"]";
    
    cout <<endl;

    cout <<"Phi_max \t";
    for(j = 0; j < num_conv_layer + num_lfc_layer; j++)
        cout<< phi_max[j].get(GRB_DoubleAttr_X) << "\t";
    cout <<endl;

    cout <<"Phi_tot \t";
    for(j = 0; j < num_conv_layer + num_lfc_layer; j++)
        cout<< phi_tot[j].get(GRB_DoubleAttr_X) << "\t";
    cout <<endl <<endl;
    
    //data dump
    cout <<"lut dump " <<endl; 
    for(k = 0; k < t; k++) {
        for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
            cout << "a["<<i<<"]";
            for(j = 0; j < num_conv_layer + num_lfc_layer; j++) {
                cout << "\t"<<a[k][i][j].get(GRB_DoubleAttr_X);
            }
            cout <<endl;
        }
            cout <<endl <<endl;

    }
    
    cout <<endl <<endl;
    //data dump
    for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
        cout << "b["<<i<<"]";
        for(j = 0; j < num_conv_layer + num_lfc_layer; j++) {
            cout << "\t"<<b[i][j].get(GRB_DoubleAttr_X);
        }
        cout <<endl;
    }

     cout <<endl <<endl;
    for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
        cout << "d["<<i<<"]";
        for(j = 0; j < num_conv_layer + num_lfc_layer; j++) {
            cout << "\t"<<d[i][j].get(GRB_DoubleAttr_X);
        }
        cout <<endl;
    }

    
/*
    for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
        cout << "b["<<i<<"]";
        for(j = 0; j < num_conv_layer + num_lfc_layer; j++) {
            cout << "\t"<<b[i][j].get(GRB_DoubleAttr_X);
        }
        cout <<endl;
    }
  */  
    cout << "****** The latency of each layer in clock cycles *****" << endl;
    for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
        cout<< "latency  "<< i << "\t" << layer_lat[i].get(GRB_DoubleAttr_X) << endl;
    }

    unsigned long long int total_lat_clock_cycles  = 0;
    float total_latency = 0.0;
 /*   for(i = 0;  i < num_conv_layer + num_lfc_layer; i++)
        total_lat_clock_cycles += layer_lat[i].get(GRB_DoubleAttr_X);
*/   
    for(j = 0; j < num_conv_layer + num_lfc_layer; j++) {
        total_lat_clock_cycles += (BATCH_SIZE - 1) * phi_max[j].get(GRB_DoubleAttr_X) + phi_tot[j].get(GRB_DoubleAttr_X);
    }

 
    //total latency is calculated using a latency calculation for a pipeline
//    total_lat_clock_cycles += (BATCH_SIZE - 1) * max_lat.get(GRB_DoubleAttr_X);

    //total_latency in ms
    total_latency = (float) (total_lat_clock_cycles / 2) / 100000;
//    cout<< "the maximum latency is "<< max_lat.get(GRB_DoubleAttr_X) <<endl;
    cout<< "total latency in clock cycles is "<< total_lat_clock_cycles <<endl;
    cout<< "total latency in ms is "<< total_latency <<endl;
    }
    else {
         
        model.set(GRB_IntParam_Threads, 8);    
        model.set(GRB_DoubleParam_TimeLimit, 120);
        model.computeIIS();

        cout<< "the following constraints can not be satisfied" <<endl;
        c = model.getConstrs();

        for(i = 0; i < model.get(GRB_IntAttr_NumConstrs); i++)
            if(c[i].get(GRB_IntAttr_IISConstr) == 1)
                cout << c[i].get(GRB_StringAttr_ConstrName) << endl;
    }

    }
        catch(GRBException e)
    {
        cout << "Error code =" << e.getErrorCode() << endl;
        cout<< e.getMessage() << endl;

        return 0;
    }
    catch (...)
    {
        cout <<"exception while solving milp" << endl;
        return 0;
    }
    cout << "done !!! "<<endl;
    
    return 0;
}
