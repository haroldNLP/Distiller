L3_attention_mse=[{"layer_T":4,  "layer_S":1, "feature":"attention", "loss":"attention_mse", "weight":1},
                  {"layer_T":8,  "layer_S":2, "feature":"attention", "loss":"attention_mse", "weight":1},
                  {"layer_T":12, "layer_S":3, "feature":"attention", "loss":"attention_mse", "weight":1}]

L3_attention_ce=[{"layer_T":0,  "layer_S":0, "feature":"attention", "loss":"attention_ce", "weight":1},
                 {"layer_T":5,  "layer_S":1, "feature":"attention", "loss":"attention_ce", "weight":1},
                 {"layer_T":11, "layer_S":2, "feature":"attention", "loss":"attention_ce", "weight":1}]

L3_hybrid=[{"layer_T":[0,0],  "layer_S":[0,0], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[1,1],  "layer_S":[1,1], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[2,2],  "layer_S":[2,2], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[3,3],"layer_S":[3,3], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":1,  "layer_S":1, "feature":"hidden", "loss":"hidden_ce", "weight":0.5},
                {"layer_T":2,  "layer_S":1, "feature":"hidden", "loss":"hidden_ce", "weight":0.5},
                {"layer_T":3, "layer_S":2, "feature":"hidden", "loss":"hidden_ce", "weight":0.5},
                {"layer_T":4,  "layer_S":2, "feature":"hidden", "loss":"hidden_ce", "weight":0.5},
                {"layer_T":5,  "layer_S":3, "feature":"hidden", "loss":"hidden_ce", "weight":0.5},
                {"layer_T":6, "layer_S":3, "feature":"hidden", "loss":"hidden_ce", "weight":0.5}
                ]


L3_attention_mse_sum=[{"layer_T":4,  "layer_S":1, "feature":"attention", "loss":"attention_mse_sum", "weight":1},
                      {"layer_T":8,  "layer_S":2, "feature":"attention", "loss":"attention_mse_sum", "weight":1},
                      {"layer_T":12, "layer_S":3, "feature":"attention", "loss":"attention_mse_sum", "weight":1}]

L3_attention_ce_mean=[{"layer_T":4,  "layer_S":1, "feature":"attention", "loss":"attention_ce_mean", "weight":1},
                      {"layer_T":8,  "layer_S":2, "feature":"attention", "loss":"attention_ce_mean", "weight":1},
                      {"layer_T":12, "layer_S":3, "feature":"attention", "loss":"attention_ce_mean", "weight":1}]


L3_hidden_smmd=[{"layer_T":[0,0],  "layer_S":[0,0], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[4,4],  "layer_S":[1,1], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[8,8],  "layer_S":[2,2], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[12,12],"layer_S":[3,3], "feature":"hidden", "loss":"mmd", "weight":1}]

L3n_hidden_mse=[{"layer_T":0, "layer_S":0, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",384,768]},
                {"layer_T":4, "layer_S":1, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",384,768]},
                {"layer_T":8, "layer_S":2, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",384,768]},
                {"layer_T":12,"layer_S":3, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",384,768]}]

L3_hidden_mse=[{"layer_T":0, "layer_S":0, "feature":"hidden", "loss":"hidden_mse", "weight":1},
               {"layer_T":4, "layer_S":1, "feature":"hidden", "loss":"hidden_mse", "weight":1},
               {"layer_T":8, "layer_S":2, "feature":"hidden", "loss":"hidden_mse", "weight":1},
               {"layer_T":12,"layer_S":3, "feature":"hidden", "loss":"hidden_mse", "weight":1}]

L3l_hidden_mse=[{"layer_T":0, "layer_S":0, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",1024,768]},
                {"layer_T":4, "layer_S":1, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",1024,768]},
                {"layer_T":8, "layer_S":2, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",1024,768]},
                {"layer_T":12,"layer_S":3, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",1024,768]}]
#######################L4################
L4_attention_mse=[{"layer_T":3,  "layer_S":1, "feature":"attention", "loss":"attention_mse", "weight":1},
                  {"layer_T":6,  "layer_S":2, "feature":"attention", "loss":"attention_mse", "weight":1},
                  {"layer_T":9,  "layer_S":3, "feature":"attention", "loss":"attention_mse", "weight":1},
                  {"layer_T":12, "layer_S":4, "feature":"attention", "loss":"attention_mse", "weight":1}]

L4_attention_ce=[{"layer_T":3,  "layer_S":1, "feature":"attention", "loss":"attention_ce", "weight":1},
                 {"layer_T":6,  "layer_S":2, "feature":"attention", "loss":"attention_ce", "weight":1},
                 {"layer_T":9,  "layer_S":3, "feature":"attention", "loss":"attention_ce", "weight":1},
                 {"layer_T":12, "layer_S":4, "feature":"attention", "loss":"attention_ce", "weight":1}]

L4_attention_ce_large=[{"layer_T":6,  "layer_S":1, "feature":"attention", "loss":"attention_ce", "weight":1},
                 {"layer_T":12,  "layer_S":2, "feature":"attention", "loss":"attention_ce", "weight":1},
                 {"layer_T":18,  "layer_S":3, "feature":"attention", "loss":"attention_ce", "weight":1},
                 {"layer_T":24, "layer_S":4, "feature":"attention", "loss":"attention_ce", "weight":1}]

L4_attention_mse_sum=[{"layer_T":3,  "layer_S":1, "feature":"attention", "loss":"attention_mse_sum", "weight":1},
                      {"layer_T":6,  "layer_S":2, "feature":"attention", "loss":"attention_mse_sum", "weight":1},
                      {"layer_T":9,  "layer_S":3, "feature":"attention", "loss":"attention_mse_sum", "weight":1},
                      {"layer_T":12, "layer_S":4, "feature":"attention", "loss":"attention_mse_sum", "weight":1}]

L4_attention_ce_mean=[{"layer_T":3,  "layer_S":1, "feature":"attention", "loss":"attention_ce_mean", "weight":1},
                      {"layer_T":6,  "layer_S":2, "feature":"attention", "loss":"attention_ce_mean", "weight":1},
                      {"layer_T":9,  "layer_S":3, "feature":"attention", "loss":"attention_ce_mean", "weight":1},
                      {"layer_T":12, "layer_S":4, "feature":"attention", "loss":"attention_ce_mean", "weight":1}]

L4_hidden_smmd=[{"layer_T":[0,0],  "layer_S":[0,0], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[3,3],  "layer_S":[1,1], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[6,6],  "layer_S":[2,2], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[9,9],  "layer_S":[3,3], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[12,12],"layer_S":[4,4], "feature":"hidden", "loss":"mmd", "weight":1}]

L4_hidden_smmd_large=[{"layer_T":[20,20],  "layer_S":[0,0], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[21,21],  "layer_S":[1,1], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[22,22],  "layer_S":[2,2], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[23,23],  "layer_S":[3,3], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[24,24],"layer_S":[4,4], "feature":"hidden", "loss":"mmd", "weight":1}]

L4t_hidden_mse=[{"layer_T":0, "layer_S":0, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",384,768]},
                {"layer_T":3, "layer_S":1, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",384,768]},
                {"layer_T":6, "layer_S":2, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",384,768]},
                {"layer_T":9, "layer_S":3, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",384,768]},
                {"layer_T":12,"layer_S":4, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",384,768]}]

L4t_hidden_mse_large=[{"layer_T":20, "layer_S":0, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",384,1024]},
                {"layer_T":21, "layer_S":1, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",384,1024]},
                {"layer_T":22, "layer_S":2, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",384,1024]},
                {"layer_T":23, "layer_S":3, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",384,1024]},
                {"layer_T":24,"layer_S":4, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",384,1024]}]

L4t_attention_ce_large=[{"layer_T":0,  "layer_S":0, "feature":"attention", "loss":"attention_ce", "weight":1},
                 {"layer_T":21,  "layer_S":1, "feature":"attention", "loss":"attention_ce", "weight":1},
                 {"layer_T":22,  "layer_S":2, "feature":"attention", "loss":"attention_ce", "weight":1},
                 {"layer_T":23, "layer_S":3, "feature":"attention", "loss":"attention_ce", "weight":1}]

L6_hidden_mse=[{"layer_T":0, "layer_S":0, "feature":"hidden", "loss":"hidden_mse", "weight":1},
               {"layer_T":2, "layer_S":1, "feature":"hidden", "loss":"hidden_mse", "weight":1},
               {"layer_T":4, "layer_S":2, "feature":"hidden", "loss":"hidden_mse", "weight":1},
               {"layer_T":6,"layer_S":3, "feature":"hidden", "loss":"hidden_mse", "weight":1},
               {"layer_T": 8, "layer_S": 4, "feature": "hidden", "loss": "hidden_mse", "weight": 1},
               {"layer_T": 10, "layer_S": 5, "feature": "hidden", "loss": "hidden_mse", "weight": 1},
               {"layer_T": 12, "layer_S": 6, "feature": "hidden", "loss": "hidden_mse", "weight": 1}
               ]

L6_hidden_smmd=[{"layer_T":[0,0],  "layer_S":[0,0], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[2,2],  "layer_S":[1,1], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[4,4],  "layer_S":[2,2], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[6,6],"layer_S":[3,3], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T": [8, 8], "layer_S": [4, 4], "feature": "hidden", "loss": "mmd", "weight": 1},
                {"layer_T": [10, 10], "layer_S": [5, 5], "feature": "hidden", "loss": "mmd", "weight": 1},
                {"layer_T": [12, 12], "layer_S": [6, 6], "feature": "hidden", "loss": "mmd", "weight": 1}
                ]

L6_attention_ce=[{"layer_T":2,  "layer_S": 1, "feature":"attention", "loss":"attention_ce", "weight":1},
                 {"layer_T":4,  "layer_S": 2, "feature":"attention", "loss":"attention_ce", "weight":1},
                 {"layer_T":6, "layer_S": 3, "feature":"attention", "loss":"attention_ce", "weight":1},
                 {"layer_T": 8, "layer_S": 4, "feature": "attention", "loss": "attention_ce", "weight": 1},
                 {"layer_T": 10, "layer_S": 5, "feature": "attention", "loss": "attention_ce", "weight": 1},
                 {"layer_T": 12, "layer_S": 6, "feature": "attention", "loss": "attention_ce", "weight": 1}
                 ]
L6_attention_ce_mean=[{"layer_T":2,  "layer_S":1, "feature":"attention", "loss":"attention_ce_mean", "weight":1},
                      {"layer_T":4,  "layer_S":2, "feature":"attention", "loss":"attention_ce_mean", "weight":1},
                      {"layer_T":6, "layer_S":3, "feature":"attention", "loss":"attention_ce_mean", "weight":1},
                      {"layer_T": 8, "layer_S": 4, "feature": "attention", "loss": "attention_ce_mean", "weight": 1},
                      {"layer_T": 10, "layer_S": 5, "feature": "attention", "loss": "attention_ce_mean", "weight": 1},
                      {"layer_T": 12, "layer_S": 6, "feature": "attention", "loss": "attention_ce_mean", "weight": 1}
                      ]




matches={'L3_attention_mse':L3_attention_mse,'L3_attention_mse_sum':L3_attention_mse_sum,
         'L3_attention_ce' :L3_attention_ce, 'L3_attention_ce_mean':L3_attention_ce_mean,
         'L3n_hidden_mse'  :L3n_hidden_mse,  'L3_hidden_smmd'      :L3_hidden_smmd,
         'L3l_hidden_mse'  :L3l_hidden_mse,  'L3_hidden_mse': L3_hidden_mse,
         'L4_attention_mse':L4_attention_mse,'L4_attention_mse_sum':L4_attention_mse_sum,
         'L4_attention_ce' :L4_attention_ce, 'L4_attention_ce_mean':L4_attention_ce_mean,
         'L4t_hidden_mse'  :L4t_hidden_mse,  'L4_hidden_smmd'      :L4_hidden_smmd,
         'L6_hidden_mse' : L6_hidden_mse, 'L6_hidden_smmd' : L6_hidden_smmd,
         'L6_attention_ce': L6_attention_ce, 'L6_attention_ce_mean': L6_attention_ce_mean,
         'L3_hybrid':L3_hybrid, 'L4_hidden_smmd_large':L4_hidden_smmd_large,
         'L4t_hidden_mse_large':L4t_hidden_mse_large, 'L4_attention_ce_large':L4_attention_ce_large,
         'L4t_attention_ce_large': L4t_attention_ce_large
        }
