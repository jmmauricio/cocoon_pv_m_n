# cocoon_pv_m_n
PV Plants benchmarks


## Example of code to get the name of each generator

    M = 2
    N = 3
    for i_m in range(1,M+1):
        for i_n in range(1,N+1):
            name = 'LV' + f"{i_m}".zfill(2) + f"{i_n}".zfill(2)