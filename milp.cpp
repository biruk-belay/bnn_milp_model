#include <iostream>
#include <vector>
#include <cmath>
#include "gurobi_c++.h"
#include "gurobi_c.h"
#include "milp.h"
#include "milp_top.h"

using namespace std;

#define BATCH_SIZE 256
typedef vector<GRBVar>          GRBVarArray;
typedef vector<GRBVarArray>     GRBVar2DArray; 

int milp_solver(unsigned int num_lut, unsigned int num_conv_layer,
                unsigned int num_lfc_layer, unsigned long long int *num_ops_per_layer,
                model_variables *res_model_lfc, model_variables *res_model_conv)
{
    int MAX_LUT_PYNQ = num_lut;

    int i, j, k;
    int status;

    unsigned int pe_th = 16; 
    unsigned int simd_th = 16; 
    unsigned long long BIG_M = 10000000000000; //represents infinity
    double episilon = 0.1;
 
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
    GRBVarArray alpha (num_conv_layer + num_lfc_layer);
        for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
            alpha[i] = model.addVar(0.0,  1.0, 0.0, GRB_INTEGER);
         }

    /**************************************************************************
         name: gamma
         type: binary
         func: gamma[i] is used to choose the interval of S   
     ***************************************************************************/
    GRBVarArray gamma (num_conv_layer + num_lfc_layer);
        for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
            gamma[i] = model.addVar(0.0,  1.0, 0.0, GRB_INTEGER);
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
         func: lut[i] represent the number or LUT in the ith layer
    ***********************************************************************/
    GRBVarArray lut(num_conv_layer + num_lfc_layer);
    for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
        lut[i] = model.addVar(1.0, MAX_LUT_PYNQ, 0.0, GRB_CONTINUOUS);
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
    GRBVarArray x (num_conv_layer + num_lfc_layer);
        for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
            x[i] = model.addVar(0.0,  1.0, 0.0, GRB_BINARY);
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

    model.update();
    /****************************************************************************
    Constr 1.1: alpha[][0] == 0 iff pe < pe_th else alpha[][0] == 1
                alpha[][1] == 0 iff simd < pe_th else alpha[][1] == 1

                alpha[][0] and alpha[][1] are used to decide which model to use
                to calculate the resource consumption
    ****************************************************************************/
    for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
        model.addConstr(P[i] - episilon >= pe_th - BIG_M * (1 - alpha[i]), "1");
        model.addConstr(P[i] - episilon <= pe_th + BIG_M * alpha[i], "2");
    }
    
    for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
        model.addConstr(S[i] - episilon >= simd_th - BIG_M * (1 - gamma[i]), "3");
        model.addConstr(S[i] - episilon <= simd_th + BIG_M * gamma[i], "4");
    }
        
    /******************************************************************
        Constr 1.2: lut[i] is the amount of LUTs in the i-th layer. 
                    This constraint enforces the amount of LUT in each 
                    layer                
    ******************************************************************/
    GRBLinExpr exp;
        for(i = 0; i < num_conv_layer; i++) {
        model.addConstr(lut[i] >= res_model_conv[0].pe_coeff   * P[i]   +
                                  res_model_conv[0].simd_coeff * S[i] +
                                  res_model_conv[0].intercept -
                                  ((alpha[i] + gamma[i]) * BIG_M), "5");

        model.addConstr(lut[i] >= res_model_conv[1].pe_coeff   * P[i]   +
                                  res_model_conv[1].simd_coeff * S[i] +
                                  res_model_conv[1].intercept -
                                  ((1 - alpha[i] + gamma[i]) * BIG_M), "6");

        model.addConstr(lut[i] >= res_model_conv[2].pe_coeff   * P[i]   +
                                  res_model_conv[2].simd_coeff * S[i] +
                                  res_model_conv[2].intercept -
                                  ((1 + alpha[i] - gamma[i]) * BIG_M), "7");

        model.addConstr(lut[i] >= res_model_conv[3].pe_coeff   * P[i]   +
                                  res_model_conv[3].simd_coeff * S[i] +
                                  res_model_conv[3].intercept -
                                  ((2 - alpha[i] - gamma[i]) * BIG_M), "8");

    }


    for(i = num_conv_layer; i < num_lfc_layer + num_conv_layer; i++) {
        model.addConstr(lut[i] >= res_model_lfc[0].pe_coeff   * P[i]   +
                                  res_model_lfc[0].simd_coeff * S[i] + 
                                  res_model_lfc[0].intercept -
                                  (alpha[i] + gamma[i]) * BIG_M, "4");
        
        model.addConstr(lut[i] >= res_model_lfc[1].pe_coeff   * P[i]   +
                                  res_model_lfc[1].simd_coeff * S[i] + 
                                  res_model_lfc[1].intercept -
                                  (1 - alpha[i] + gamma[i]) * BIG_M, "5");
        
        model.addConstr(lut[i] >= res_model_lfc[2].pe_coeff   * P[i]   +
                                  res_model_lfc[2].simd_coeff * S[i] +
                                  res_model_lfc[2].intercept -
                                  (1 + alpha[i] - gamma[i]) * BIG_M, "6");

        model.addConstr(lut[i] >= res_model_lfc[3].pe_coeff   * P[i]   +
                                  res_model_lfc[3].simd_coeff * S[i] +
                                  res_model_lfc[3].intercept -
                                  (2 - alpha[i] - gamma[i]) * BIG_M, "7");
    
    }
    
    /******************************************************************
        Constr 1.3: This is a mirror of constr 1.2 to enforce
                    equality
    ******************************************************************/

    for(i = 0; i < num_conv_layer; i++) {
        model.addConstr((res_model_conv[0].pe_coeff  *  P[i]   +
                        res_model_conv[0].simd_coeff *  S[i] + 
                        res_model_conv[0].intercept)  >= (lut[i]  - 
                        ((alpha[i] + gamma[i]) * BIG_M)), "9");
        
        model.addConstr((res_model_conv[1].pe_coeff  * P[i]    +
                        res_model_conv[1].simd_coeff * S[i]  + 
                        res_model_conv[1].intercept) >= (lut[i]   -
                        (1 - alpha[i] + gamma[i]) * BIG_M), "10");
        
        model.addConstr((res_model_conv[2].pe_coeff  * P[i]    +
                        res_model_conv[2].simd_coeff * S[i]  +
                        res_model_conv[2].intercept)  >= (lut[i]  -      
                            ((1 + alpha[i] - gamma[i]) * BIG_M)), "11");

        model.addConstr((res_model_conv[3].pe_coeff  * P[i]   +
                        res_model_conv[3].simd_coeff * S[i] +
                        res_model_conv[3].intercept) >= (lut[i]  -
                        ((2 - alpha[i] - gamma[i]) * BIG_M)), "12");
   
    }
     

    for(i = num_conv_layer; i < num_lfc_layer + num_conv_layer; i++) {
        model.addConstr(res_model_lfc[0].pe_coeff   *  P[i]   +
                        res_model_lfc[0].simd_coeff *  S[i] + 
                        res_model_lfc[0].intercept  >= lut[i]  - 
                        (alpha[i] + gamma[i]) * BIG_M, "9");
        
        model.addConstr(res_model_lfc[1].pe_coeff   * P[i]    +
                        res_model_lfc[1].simd_coeff * S[i]  + 
                        res_model_lfc[1].intercept >= lut[i]   -
                        (1 - alpha[i] + gamma[i]) * BIG_M, "10");
        
        model.addConstr(res_model_lfc[2].pe_coeff   * P[i]    +
                        res_model_lfc[2].simd_coeff * S[i]  +
                        res_model_lfc[2].intercept  >= lut[i]  - 
                                  (1 + alpha[i] - gamma[i]) * BIG_M, "11");

        model.addConstr(res_model_lfc[3].pe_coeff   * P[i]   +
                        res_model_lfc[3].simd_coeff * S[i] +
                        res_model_lfc[3].intercept >= lut[i]  -
                        (2 - alpha[i] - gamma[i]) * BIG_M, "12");
    
    }
  
    for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
        exp += lut[i];
    }

    model.addConstr(exp <= MAX_LUT_PYNQ, "8");
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
        exp6 += 3 * (j+1) * delta[0][j];
        exp6 += 3 * 1 * delta[0][j];
        exp5 += delta[0][j];
    }       
 
    //Make simd[0] always 0   
//    model.addConstr(simd[0] == exp6, "125");     
    model.addConstr(exp5 == 1, "126");
    model.addConstr(S[0] == 3);
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
            //exp8 += 3 * (i+1) * pow(2, j+1) * beta_tau[0][i * SIMD_EXP + j];
            exp8 += 3 * 1 * pow(2, j+1) * lambda[0][i * SIMD_EXP + j];
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
            //model.addConstr(tau[0] >= 3 * (j+1) * pe[0] - (1 - beta_simd[0][j]) * BIG_M, "171");
            model.addConstr(tau[0] >= 3 * 1 * P[0] - (1 - delta[0][j]) * BIG_M, "171");
            //model.addConstr(3 * (j+1) * pe[0] >= tau[0] - (1 - beta_simd[0][j]) * BIG_M, "181");
            model.addConstr(3 * 1 * P[0] >= tau[0] - (1 - delta[0][j]) * BIG_M, "181");
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
         //model.addConstr((num_ops_per_layer[0] / (3 * (i+1) * pow(2, j+1))) >=  layer_lat[0] - (1 - beta_tau[0][i * SIMD_EXP + j]) * BIG_M, "19");
         model.addConstr((num_ops_per_layer[0] / (3 * pow(2, j+1))) >=  layer_lat[0] - (1 - lambda[0][i * SIMD_EXP + j]) * BIG_M, "19");
         //model.addConstr(layer_lat[0]  >= (num_ops_per_layer[0] / (3 * (i + 1) * pow(2, j+1))) - (1 - beta_tau[0][i * SIMD_EXP + j]) * BIG_M, "20");
         model.addConstr(layer_lat[0]  >= (num_ops_per_layer[0] / (3 * 1 * pow(2, j+1))) - (1 - lambda[0][i * SIMD_EXP + j]) * BIG_M, "20");
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

    /*
    GRBLinExpr exp4;
    for(i = 0;  i < num_conv_layer + num_lfc_layer; i++)
        exp4 += layer_lat[i];
    */ 

    //Updated delay modeling   
    GRBLinExpr exp4, exp3; 
    for(i = 0;  i < num_conv_layer + num_lfc_layer; i++)
        model.addConstr(max_lat >= layer_lat[i], "121");
        
    for(i = 0;  i < num_conv_layer + num_lfc_layer; i++)
        exp3 += layer_lat[i];
            
    exp4 = (BATCH_SIZE - 1) * max_lat + exp3;
    
    /**********************************************************************
        Objective function: minimize the latency
    **********************************************************************/
    model.setObjective(exp4,  GRB_MINIMIZE);

    //start optimization
    model.optimize();
    
    status = model.get(GRB_IntAttr_Status);
    
    if(status == GRB_OPTIMAL) {

    /**********************************************************************
                print outputs
    **********************************************************************/
    
    int total_luts_used = 0;
    for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
        total_luts_used += lut[i].get(GRB_DoubleAttr_X);
        cout << " pe " << P[i].get(GRB_DoubleAttr_X) << "\tsimd " << S[i].get(GRB_DoubleAttr_X) << 
             "\t alpha " << alpha[i].get(GRB_DoubleAttr_X) << "\t" << gamma[i].get(GRB_DoubleAttr_X) << "\t lut " <<
              lut[i].get(GRB_DoubleAttr_X) << endl;
    }

    cout<< endl;
    
    cout << " total luts used " << total_luts_used << endl;

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
#if defined(FIRST_CUT) || defined(FULL)
    cout<< "PE * SIMD \t" << tau[0].get(GRB_DoubleAttr_X); 
    for(i = 0; i < SIMD_EXP; i++) {
        for(j = 0; j < PE_EXP; j++)
            cout << "\t" << lambda[0][i * SIMD_EXP + j].get(GRB_DoubleAttr_X);
    }
    cout <<endl;
    for(i = 1; i < num_conv_layer + num_lfc_layer; i++) {
        cout<< "PE * SIMD \t" << tau[i].get(GRB_DoubleAttr_X);
        for(j = 0; j < SIMD_EXP + PE_EXP; j++)
            cout << "\t" << lambda[i][j].get(GRB_DoubleAttr_X);
        cout <<endl;
    }

#else     
    for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
        cout<< "PE * SIMD \t" << tau[i].get(GRB_DoubleAttr_X);
        for(j = 0; j < SIMD_EXP + PE_EXP; j++)
            cout << "\t" << lambda[i][j].get(GRB_DoubleAttr_X);
        cout <<endl;
    }
    cout <<endl;
#endif

    cout << "****** The latency of each layer in clock cycles *****" << endl;
    for(i = 0; i < num_conv_layer + num_lfc_layer; i++) {
        cout<< "latency  "<< i << "\t" << layer_lat[i].get(GRB_DoubleAttr_X) << endl;
    }

    unsigned long long int total_lat_clock_cycles  = 0;
    float total_latency = 0.0;
    for(i = 0;  i < num_conv_layer + num_lfc_layer; i++)
        total_lat_clock_cycles += layer_lat[i].get(GRB_DoubleAttr_X);
    
    //total latency is calculated using a latency calculation for a pipeline
    total_lat_clock_cycles += (BATCH_SIZE - 1) * max_lat.get(GRB_DoubleAttr_X);

    //total_latency in ms
    total_latency = (float) (total_lat_clock_cycles / 2) / 100000;
    cout<< "the maximum latency is "<< max_lat.get(GRB_DoubleAttr_X) <<endl;
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
