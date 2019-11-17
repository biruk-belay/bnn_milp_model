typedef struct {
    double pe_coeff;
    double simd_coeff;
    double intercept;
}model_variables;

typedef struct {
    int IFM;
    int IFM_CH;
    int OFM;
    int OFM_CH;
    int k;
}conv_ntwk_arch;

typedef struct {
    int matW;
    int matH;
}lfc_ntwk_arch;


int milp_solver(unsigned int num_lut, unsigned int num_conv_layer,
                           unsigned int num_lfc_layer, unsigned long long *num_ops_per_layer,
                           model_variables *res_model_lfc, model_variables *res_model_conv);

int milp_solver_first_cut(unsigned int num_lut, unsigned int num_conv_layer,
                           unsigned int num_lfc_layer, unsigned long long int *num_ops_per_layer,
                           model_variables *res_model_lfc, model_variables *res_model_conv);

int milp_solver_full(unsigned int num_lut, unsigned int num_conv_layer,
                           unsigned int num_lfc_layer, unsigned long long int *num_ops_per_layer,
                           model_variables *res_model_lfc, model_variables *res_model_conv);
//void milp_solver_full(float res_percentage);
//void milp_solver_first_cut(float res_percentage); 
