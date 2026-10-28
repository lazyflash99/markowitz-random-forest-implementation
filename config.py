# Central configuration for datasets, models, and optimization parameters.
CONFIG = {
    "N_TRIALS": 20,         
    "TPE_EVALS": 20,        
    "N_EST_MAX": 200,       
    "XGB_N_EST": 100,       
    "LAMBDA": 0.25,         #PyPortfolioOpt max_quadratic_utility(risk_aversion=0.5): 0.5 * 0.5 = 0.25
    "L2_GAMMA": 0.1,        #inverse L2 regularisation of weights
    "SEED": 0,
    "TEST_SIZE": 0.2,  #80/20 split
    "RESULTS_DIR": "results",
    "MAX_ROWS": {},
    #the 2 datasets per task:
    "DATASETS": {
        "shoppers":          {"task": "binary",     "id": 468},
        "bank_marketing":    {"task": "binary",     "id": 222},
        "wine":              {"task": "regression", "id": 186},
        "superconductivity": {"task": "regression", "id": 464},
        "pen":               {"task": "multiclass", "id": 81},
        "student":           {"task": "multiclass", "id": 697},
    },
    "GLOBAL_PREPROCESSING": True,    #global deduplication, imputation, and encoding before splitting
    "MODEL_AGGREGATION": True,       #aggregate 3-CV models for RF/ERT
    "XGB_BALANCED_WEIGHTS": False,
}
PRIMARY = {  # primary metric per task and whether higher is better
    "binary": ("PRAUC", True),
    "regression": ("MAE", False),
    "multiclass": ("F1_macro", True),
}

