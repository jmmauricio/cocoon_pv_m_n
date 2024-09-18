import json



with open('pv_2_3.json','r') as fobj:
    data_dict = json.loads(fobj.read())


S_base = data_dict['system']['S_base']

for line in data_dict['lines']:
    S_mva = line['S_mva']
    R_pu = line['R_pu']*S_base/(S_mva*1e6)
    line['R_pu'] = R_pu
    X_pu = line['X_pu']*S_base/(S_mva*1e6)
    line['X_pu'] = X_pu
    line['S_mva'] = S_base/1e6

for trafo in data_dict['transformers']:
    S_mva = trafo['S_mva']
    R_pu = trafo['R_pu']*S_base/(S_mva*1e6)
    trafo['R_pu'] = R_pu
    X_pu = trafo['X_pu']*S_base/(S_mva*1e6)
    trafo['X_pu'] = X_pu
    trafo['S_mva'] = S_base/1e6

with open('pv_2_3_pu.json','w') as fobj:
    fobj.write(json.dumps(data_dict, indent=2))
