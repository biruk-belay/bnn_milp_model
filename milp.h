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

conv_ntwk_arch conv_net[] = {{32, 3, 30, 64, 3},
                             {30, 64, 28, 64, 3},
                             {14, 64, 12, 128, 3},
                             {12, 128, 10, 128, 3},
                             {5, 128, 3, 256, 3},
                             {3, 256, 1, 256, 3}};

lfc_ntwk_arch lfc_net[] = {{256, 512}, {512, 512},
                           {512, 64}};

