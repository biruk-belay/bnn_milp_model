#include <iostream>
#include <vector>
#include <cmath>
#include "gurobi_c++.h"
#include "gurobi_c.h"
#include "milp.h"

using namespace std;

#define PE_MAX 512
#define SIMD_MAX 512
#define PE_EXP 9
#define SIMD_EXP 9
#define MAX_LUT_PYNQ 55637
#define NUM_QUADRANTS 4

typedef vector<GRBVar>          GRBVarArray;
typedef vector<GRBVarArray>     GRBVar2DArray;

unsigned int num_layer = 4;
unsigned int pe_threshold = 16; //
unsigned int simd_threshold = 16; //
int BIG_M = 1000000; //represents infinity
double episilon = 0.1;

/***********************************************************
    network_arch is the number of operations per layer
    network_arch[0] = 842  * 1024
    network_arch[1] = 1024 * 1024
    network_arch[2] = 1024 * 1024
    network_arch[3] = 1024 * 64
********************************************************/
unsigned int network_arch[] = {862208, 1048576, 1048576, 65536};

int milp_solver()
{
    /*********************************************************************************
        #LUT(i) = 97.24 * pe(i) + 25.52 * simd(i) + 4110 -- quadrant 1
        #LUT(i) = 88.72 * pe(i) + 122   * simd(i) + 3417 -- quadrant 2
        #LUT(i) = 242.1 * pe(i) + 209.4 * simd(i) - 3380 -- quadrant 3
        #LUT(i) = 250   * pe(i) + 45.94 * simd(i) + 2544 -- quadrant 4
    ******************************************************************************/
    model_variables res_model[NUM_QUADRANTS] = {{97.24, 28.52, 4110}, {88.72, 122, 3417},
                                                {242.1, 209.4, -3380}, {250, 45.94, 2544}};
    int i, j, k;
    try {
    
        GRBEnv env = GRBEnv();
        GRBConstr* c = NULL;
        GRBModel model = GRBModel(env);

    /**********************************************************************
         name: pe
         type: integer
         func: pe[i] represent the PE in each layer
     ***********************************************************************/    
    GRBVarArray pe(num_layer);
    for(i = 0; i < num_layer; i++) {
        pe[i] = model.addVar(2.0, PE_MAX, 0.0, GRB_INTEGER);
    }

    /**********************************************************************
        name: simd
         type: integer
         func: simd[i] represent the simd in each neuron on a single layer
    ***********************************************************************/

    GRBVarArray simd(num_layer);
    for(i = 0; i < num_layer; i++) {
        simd[i] = model.addVar(2.0, SIMD_MAX, 0.0, GRB_INTEGER);
    }

    /**************************************************************************
         name: alpha
         type: binary
         func: alpha[i][0] and alpha[i][1] are used for determining the model to 
               choose to calculate the resource consumption. In this implementation 
               they represent the quadrants of the resource model to choose from.   
     ***************************************************************************/
    GRBVar2DArray alpha (num_layer);
        for(i = 0; i < num_layer; i++) {
            GRBVarArray each_alpha(2);
            alpha[i] = each_alpha;
            for(k = 0; k < 2; k++)
                alpha[i][k] = model.addVar(0.0,  1.0, 0.0, GRB_INTEGER);
         }

     /**************************************************************************
         name: beta_pe
         type: binary
         func: beta_pe[i][0] and beta_pe[i][1]
     ***************************************************************************/
    GRBVar2DArray beta_pe (num_layer);
        for(i = 0; i < num_layer; i++) {
            GRBVarArray each_alpha(PE_MAX);
            beta_pe[i] = each_alpha;
            for(k = 0; k < PE_MAX; k++)
                beta_pe[i][k] = model.addVar(0.0,  1.0, 0.0, GRB_BINARY);
         }

     /**************************************************************************
         name: beta_simd
         type: binary
         func: beta_simd[i][0] and beta_simd[i][1]
     ***************************************************************************/
    GRBVar2DArray beta_simd(num_layer);
        for(i = 0; i < num_layer; i++) {
            GRBVarArray each_alpha(SIMD_MAX);
            beta_simd[i] = each_alpha;
            for(k = 0; k < SIMD_MAX; k++)
                beta_simd[i][k] = model.addVar(0.0,  1.0, 0.0, GRB_BINARY);
         }
    /**********************************************************************
         name: lat
         type: integer
         func: lat represent the total latency of the neural net
     ***********************************************************************/
    GRBVar lat = model.addVar(1.0, GRB_INFINITY, 0.0, GRB_INTEGER);

    /**********************************************************************
         name: lut
         type: integer
         func: lut[i] represent the number or LUT in the ith layer
    ***********************************************************************/
    GRBVarArray lut(num_layer);
    for(i = 0; i < num_layer; i++) {
        lut[i] = model.addVar(0.0, MAX_LUT_PYNQ, 0.0, GRB_INTEGER);
    }
   
   /**********************************************************************
         name: tau
         type: integer
         func: lut[i] represent the number or LUT in the ith layer
    ***********************************************************************/
    GRBVarArray tau(num_layer);
    for(i = 0; i < num_layer; i++) {
        tau[i] = model.addVar(0.0, 65536, 0.0, GRB_INTEGER);
    }

    /**************************************************************************
         name: beta_tau
         type: binary
         func: beta_tau[i][0] 
     ***************************************************************************/
    GRBVar2DArray beta_tau(num_layer);
        for(i = 0; i < num_layer; i++) {
            GRBVarArray each_alpha(SIMD_EXP + PE_EXP);
            beta_tau[i] = each_alpha;
            for(k = 0; k < (SIMD_EXP + PE_EXP); k++)
                beta_tau[i][k] = model.addVar(0.0,  1.0, 0.0, GRB_BINARY);
         }

    /**************************************************************************
         name: layer_lat
         type: binary
         func: layer_lat[i] 
     ***************************************************************************/
    GRBVarArray layer_lat(num_layer);
    for(i = 0; i < num_layer; i++) {
        layer_lat[i] = model.addVar(0.0, 262144, 0.0, GRB_INTEGER);
    }
 
    model.update();

    /****************************************************************************
    Constr 1.1: alpha[][0] == 0 iff pe < pe_threshold else alpha[][0] == 1
                alpha[][1] == 0 iff simd < pe_threshold else alpha[][1] == 1

                alpha[][0] and alpha[][1] are used to decide which model to use
                to calculate the resource consumption
    ****************************************************************************/
    for(i = 0; i < num_layer; i++) {
        model.addConstr(pe[i] + episilon >= pe_threshold - BIG_M * (1 - alpha[i][0]), "1");
        model.addConstr(pe[i] + episilon <= pe_threshold + BIG_M * alpha[i][0], "2");
    }
    
    for(i = 0; i < num_layer; i++) {
        model.addConstr(simd[i] + episilon >= simd_threshold - BIG_M * (1 - alpha[i][1]), "3");
        model.addConstr(simd[i] + episilon <= simd_threshold + BIG_M * alpha[i][1], "4");
    }
        
    /******************************************************************
        Constr 1.2: lut[i] is the amount of LUTs in the i-th layer. 
                    This constraint enforces the amount of LUT in each 
                    layer                
    ******************************************************************/
    GRBLinExpr exp;
    for(i = 0; i < num_layer; i++) {
        model.addConstr(lut[i] >= res_model[0].pe_coeff   * pe[i]   +
                                  res_model[0].simd_coeff * simd[i] + 
                                  res_model[0].intercept -
                                  (alpha[i][0] + alpha[i][1]) * BIG_M, "4");
        
        model.addConstr(lut[i] >= res_model[1].pe_coeff   * pe[i]   +
                                  res_model[1].simd_coeff * simd[i] + 
                                  res_model[1].intercept -
                                  (1 - alpha[i][0] + alpha[i][1]) * BIG_M, "5");
        
        model.addConstr(lut[i] >= res_model[2].pe_coeff   * pe[i]   +
                                  res_model[2].simd_coeff * simd[i] +
                                  res_model[2].intercept -
                                  (1 + alpha[i][0] - alpha[i][1]) * BIG_M, "6");

        model.addConstr(lut[i] >= res_model[3].pe_coeff   * pe[i]   +
                                  res_model[3].simd_coeff * simd[i] +
                                  res_model[3].intercept -
                                  (2 - alpha[i][0] - alpha[i][1]) * BIG_M, "7");
    
        exp += lut[i];
    }

    model.addConstr(exp <= MAX_LUT_PYNQ, "8");
    
    /******************************************************************
        Constr 1.3: This is a mirror of constr 1.2 to enforce
                    equality
    ******************************************************************/
/*
    for(i = 0; i < num_layer; i++) {
        model.addConstr(res_model[0].pe_coeff   *  pe[i]   +
                        res_model[0].simd_coeff *  simd[i] + 
                        res_model[0].intercept  >= lut[i]  - 
                        (alpha[i][0] + alpha[i][1]) * BIG_M, "9");
        
        model.addConstr(res_model[1].pe_coeff   * pe[i]    +
                        res_model[1].simd_coeff * simd[i]  + 
                        res_model[1].intercept >= lut[i]   -
                        (1 - alpha[i][0] + alpha[i][1]) * BIG_M, "10");
        
        model.addConstr(res_model[2].pe_coeff   * pe[i]    +
                        res_model[2].simd_coeff * simd[i]  +
                        res_model[2].intercept  >= lut[i]  - 
                                  (1 + alpha[i][0] - alpha[i][1]) * BIG_M, "11");

        model.addConstr(res_model[3].pe_coeff   * pe[i]   +
                        res_model[3].simd_coeff * simd[i] +
                        res_model[3].intercept >= lut[i]  -
                        (2 - alpha[i][0] - alpha[i][1]) * BIG_M, "12");
    
    }
  */   
   /**********************************************************************
        Constr 1.4: The number of PE must be greater than SIMD in a layer
                    Part of a FINN constraint
    **********************************************************************/
    for(i = 0; i < num_layer; i++) {
        model.addConstr(pe[i] >= simd[i], "13");
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
    for(i = 0; i < num_layer; i++) {
        GRBLinExpr exp1, exp2;
        for(j = 0; j < PE_EXP; j++) {
            exp2 += pow(2, j+1) * beta_pe[i][j];
            exp1 += beta_pe[i][j];
        }
            model.addConstr(pe[i] == exp2, "15");
            model.addConstr(exp1 == 1, "16");
    }

   /**********************************************************************
        Constr 1.6: similar to constraint 1.5 but for simd
    **********************************************************************/
    for(i = 0; i < num_layer; i++) {
        GRBLinExpr exp5, exp6;
        for(j = 0; j < SIMD_EXP; j++) {
            exp6 += pow(2, j+1) * beta_simd[i][j];
            exp5 += beta_simd[i][j];
        }
            model.addConstr(simd[i] == exp6, "15");
            model.addConstr(exp5 == 1, "16");
    }

   /**********************************************************************
        Constr 1.7: similar to constr 1.5 but for tau
    **********************************************************************/
    for(i = 0; i < num_layer; i++) {
        GRBLinExpr exp5, exp6;
        for(j = 0; j < SIMD_EXP + PE_EXP; j++) {
            exp6 += pow(2, j+1) * beta_tau[i][j];
            exp5 += beta_tau[i][j];
        }
            model.addConstr(tau[i] == exp6, "15");
            model.addConstr(exp5 == 1, "16");
    }

    /**********************************************************************
        Constr 1.8: latency constraint
    **********************************************************************/
    GRBLinExpr exp3;
    for(i = 0; i < num_layer; i++) {
        for(j = 0; j < SIMD_EXP; j++) {
            model.addConstr(tau[i] >= pow(2, j+1) * pe[i] - (1 - beta_simd[i][j]) * BIG_M, "17");
            model.addConstr(pow(2, j+1) * pe[i] >= tau[i] - (1 - beta_simd[i][j]) * BIG_M, "18");
        }
       //exp3 += tau[i];
    }
    
    for(i = 0; i < num_layer; i++) {
        GRBLinExpr exp4;
        for(j = 0; j < SIMD_EXP + PE_EXP; j++) {
            model.addConstr((network_arch[i] / pow(2, j+1)) >=  layer_lat[i] - (1 - beta_tau[i][j]) * BIG_M, "17");    
            model.addConstr(layer_lat[i] >= (network_arch[i] / pow(2, j+1))- (1 - beta_tau[i][j]) * BIG_M, "18");
        }
    }
    
    GRBLinExpr exp4;
    for(i = 0;  i < num_layer; i++)
        exp4 += layer_lat[i];
    
    /**********************************************************************
        Objective function: minimize the latency
    **********************************************************************/
    model.setObjective(exp4,  GRB_MINIMIZE);

    //start optimization
    model.optimize();

    /**********************************************************************
                print outputs
    **********************************************************************/
    
    int total_luts_used = 0;
    for(i = 0; i < num_layer; i++) {
        total_luts_used += lut[i].get(GRB_DoubleAttr_X);
        cout << " pe " << pe[i].get(GRB_DoubleAttr_X) << "\tsimd " << simd[i].get(GRB_DoubleAttr_X) << 
             "\t alpha " << alpha[i][0].get(GRB_DoubleAttr_X) << "\t" << alpha[i][1].get(GRB_DoubleAttr_X) << "\t lut " <<
              lut[i].get(GRB_DoubleAttr_X) << endl;
    }

    cout<< endl;
    
    cout << " total luts used " << total_luts_used << endl;

    cout<< "pe_simd" <<endl;
    for(i = 0; i < num_layer; i++){
        cout << "layer " << i << "\t";
        for(j = 0; j < PE_EXP; j++)
            cout << "\t" << beta_pe[i][j].get(GRB_DoubleAttr_X);
        cout <<endl;
    }
    
    cout<< "beta_simd" <<endl;
    int latency = 0;
    for(i = 0; i < num_layer; i++){
        latency += tau[i].get(GRB_DoubleAttr_X);
        cout << "layer " << i << "\t";
        for(j = 0; j < SIMD_EXP; j++)
            cout << "\t" << beta_simd[i][j].get(GRB_DoubleAttr_X);
        cout <<endl;
    }
    
    cout << endl; 
    for(i = 0; i < num_layer; i++) {
        cout<< "lat \t" << tau[i].get(GRB_DoubleAttr_X); 
        for(j = 0; j < SIMD_EXP + PE_EXP; j++)
            cout << "\t" << beta_tau[i][j].get(GRB_DoubleAttr_X);
        cout <<endl;
    }
    
    cout <<endl;
    for(i = 0; i < num_layer; i++) {
        cout<< "latency  "<< i << "\t" << layer_lat[i].get(GRB_DoubleAttr_X) << endl;
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

int main()
{
    milp_solver();
    
    return 0;
}
