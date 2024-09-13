import numpy as np
import numba
import scipy.optimize as sopt
import scipy.sparse as sspa
from scipy.sparse.linalg import spsolve,spilu,splu
from numba import cuda
import cffi
import numba.core.typing.cffi_utils as cffi_support
from io import BytesIO
import pkgutil
import os

dae_file_mode = 'local'

ffi = cffi.FFI()

if dae_file_mode == 'local':
    import pv_4_5_cffi as jacs
if dae_file_mode == 'enviroment':
    import envus.no_enviroment.pv_4_5_cffi as jacs
if dae_file_mode == 'colab':
    import pv_4_5_cffi as jacs
if dae_file_mode == 'testing':
    from pydae.temp import pv_4_5_cffi as jacs
    
cffi_support.register_module(jacs)
f_ini_eval = jacs.lib.f_ini_eval
g_ini_eval = jacs.lib.g_ini_eval
f_run_eval = jacs.lib.f_run_eval
g_run_eval = jacs.lib.g_run_eval
h_eval  = jacs.lib.h_eval

de_jac_ini_xy_eval = jacs.lib.de_jac_ini_xy_eval
de_jac_ini_up_eval = jacs.lib.de_jac_ini_up_eval
de_jac_ini_num_eval = jacs.lib.de_jac_ini_num_eval

sp_jac_ini_xy_eval = jacs.lib.sp_jac_ini_xy_eval
sp_jac_ini_up_eval = jacs.lib.sp_jac_ini_up_eval
sp_jac_ini_num_eval = jacs.lib.sp_jac_ini_num_eval

de_jac_run_xy_eval = jacs.lib.de_jac_run_xy_eval
de_jac_run_up_eval = jacs.lib.de_jac_run_up_eval
de_jac_run_num_eval = jacs.lib.de_jac_run_num_eval

sp_jac_run_xy_eval = jacs.lib.sp_jac_run_xy_eval
sp_jac_run_up_eval = jacs.lib.sp_jac_run_up_eval
sp_jac_run_num_eval = jacs.lib.sp_jac_run_num_eval

de_jac_trap_xy_eval= jacs.lib.de_jac_trap_xy_eval            
de_jac_trap_up_eval= jacs.lib.de_jac_trap_up_eval        
de_jac_trap_num_eval= jacs.lib.de_jac_trap_num_eval

sp_jac_trap_xy_eval= jacs.lib.sp_jac_trap_xy_eval            
sp_jac_trap_up_eval= jacs.lib.sp_jac_trap_up_eval        
sp_jac_trap_num_eval= jacs.lib.sp_jac_trap_num_eval

sp_Fu_run_up_eval = jacs.lib.sp_Fu_run_up_eval
sp_Gu_run_up_eval = jacs.lib.sp_Gu_run_up_eval
sp_Hx_run_up_eval = jacs.lib.sp_Hx_run_up_eval
sp_Hy_run_up_eval = jacs.lib.sp_Hy_run_up_eval
sp_Hu_run_up_eval = jacs.lib.sp_Hu_run_up_eval
sp_Fu_run_xy_eval = jacs.lib.sp_Fu_run_xy_eval
sp_Gu_run_xy_eval = jacs.lib.sp_Gu_run_xy_eval
sp_Hx_run_xy_eval = jacs.lib.sp_Hx_run_xy_eval
sp_Hy_run_xy_eval = jacs.lib.sp_Hy_run_xy_eval
sp_Hu_run_xy_eval = jacs.lib.sp_Hu_run_xy_eval



import json

sin = np.sin
cos = np.cos
atan2 = np.arctan2
sqrt = np.sqrt 
sign = np.sign 
exp = np.exp


class model: 

    def __init__(self,matrices_folder='./build'): 
        
        self.matrices_folder = matrices_folder
        
        self.dae_file_mode = 'local'
        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 86
        self.N_y = 238 
        self.N_z = 156 
        self.N_store = 100000 
        self.params_list = ['S_base', 'g_POI_GRID', 'b_POI_GRID', 'bs_POI_GRID', 'g_BESS_POI_MV', 'b_BESS_POI_MV', 'bs_BESS_POI_MV', 'g_LV0101_MV0101', 'b_LV0101_MV0101', 'bs_LV0101_MV0101', 'g_MV0101_POI_MV', 'b_MV0101_POI_MV', 'bs_MV0101_POI_MV', 'g_LV0102_MV0102', 'b_LV0102_MV0102', 'bs_LV0102_MV0102', 'g_MV0102_MV0101', 'b_MV0102_MV0101', 'bs_MV0102_MV0101', 'g_LV0103_MV0103', 'b_LV0103_MV0103', 'bs_LV0103_MV0103', 'g_MV0103_MV0102', 'b_MV0103_MV0102', 'bs_MV0103_MV0102', 'g_LV0104_MV0104', 'b_LV0104_MV0104', 'bs_LV0104_MV0104', 'g_MV0104_MV0103', 'b_MV0104_MV0103', 'bs_MV0104_MV0103', 'g_LV0105_MV0105', 'b_LV0105_MV0105', 'bs_LV0105_MV0105', 'g_MV0105_MV0104', 'b_MV0105_MV0104', 'bs_MV0105_MV0104', 'g_LV0201_MV0201', 'b_LV0201_MV0201', 'bs_LV0201_MV0201', 'g_MV0201_POI_MV', 'b_MV0201_POI_MV', 'bs_MV0201_POI_MV', 'g_LV0202_MV0202', 'b_LV0202_MV0202', 'bs_LV0202_MV0202', 'g_MV0202_MV0201', 'b_MV0202_MV0201', 'bs_MV0202_MV0201', 'g_LV0203_MV0203', 'b_LV0203_MV0203', 'bs_LV0203_MV0203', 'g_MV0203_MV0202', 'b_MV0203_MV0202', 'bs_MV0203_MV0202', 'g_LV0204_MV0204', 'b_LV0204_MV0204', 'bs_LV0204_MV0204', 'g_MV0204_MV0203', 'b_MV0204_MV0203', 'bs_MV0204_MV0203', 'g_LV0205_MV0205', 'b_LV0205_MV0205', 'bs_LV0205_MV0205', 'g_MV0205_MV0204', 'b_MV0205_MV0204', 'bs_MV0205_MV0204', 'g_LV0301_MV0301', 'b_LV0301_MV0301', 'bs_LV0301_MV0301', 'g_MV0301_POI_MV', 'b_MV0301_POI_MV', 'bs_MV0301_POI_MV', 'g_LV0302_MV0302', 'b_LV0302_MV0302', 'bs_LV0302_MV0302', 'g_MV0302_MV0301', 'b_MV0302_MV0301', 'bs_MV0302_MV0301', 'g_LV0303_MV0303', 'b_LV0303_MV0303', 'bs_LV0303_MV0303', 'g_MV0303_MV0302', 'b_MV0303_MV0302', 'bs_MV0303_MV0302', 'g_LV0304_MV0304', 'b_LV0304_MV0304', 'bs_LV0304_MV0304', 'g_MV0304_MV0303', 'b_MV0304_MV0303', 'bs_MV0304_MV0303', 'g_LV0305_MV0305', 'b_LV0305_MV0305', 'bs_LV0305_MV0305', 'g_MV0305_MV0304', 'b_MV0305_MV0304', 'bs_MV0305_MV0304', 'g_LV0401_MV0401', 'b_LV0401_MV0401', 'bs_LV0401_MV0401', 'g_MV0401_POI_MV', 'b_MV0401_POI_MV', 'bs_MV0401_POI_MV', 'g_LV0402_MV0402', 'b_LV0402_MV0402', 'bs_LV0402_MV0402', 'g_MV0402_MV0401', 'b_MV0402_MV0401', 'bs_MV0402_MV0401', 'g_LV0403_MV0403', 'b_LV0403_MV0403', 'bs_LV0403_MV0403', 'g_MV0403_MV0402', 'b_MV0403_MV0402', 'bs_MV0403_MV0402', 'g_LV0404_MV0404', 'b_LV0404_MV0404', 'bs_LV0404_MV0404', 'g_MV0404_MV0403', 'b_MV0404_MV0403', 'bs_MV0404_MV0403', 'g_LV0405_MV0405', 'b_LV0405_MV0405', 'bs_LV0405_MV0405', 'g_MV0405_MV0404', 'b_MV0405_MV0404', 'bs_MV0405_MV0404', 'g_cc_POI_MV_POI', 'b_cc_POI_MV_POI', 'tap_POI_MV_POI', 'ang_POI_MV_POI', 'U_POI_MV_n', 'U_POI_n', 'U_GRID_n', 'U_BESS_n', 'U_LV0101_n', 'U_MV0101_n', 'U_LV0102_n', 'U_MV0102_n', 'U_LV0103_n', 'U_MV0103_n', 'U_LV0104_n', 'U_MV0104_n', 'U_LV0105_n', 'U_MV0105_n', 'U_LV0201_n', 'U_MV0201_n', 'U_LV0202_n', 'U_MV0202_n', 'U_LV0203_n', 'U_MV0203_n', 'U_LV0204_n', 'U_MV0204_n', 'U_LV0205_n', 'U_MV0205_n', 'U_LV0301_n', 'U_MV0301_n', 'U_LV0302_n', 'U_MV0302_n', 'U_LV0303_n', 'U_MV0303_n', 'U_LV0304_n', 'U_MV0304_n', 'U_LV0305_n', 'U_MV0305_n', 'U_LV0401_n', 'U_MV0401_n', 'U_LV0402_n', 'U_MV0402_n', 'U_LV0403_n', 'U_MV0403_n', 'U_LV0404_n', 'U_MV0404_n', 'U_LV0405_n', 'U_MV0405_n', 'K_p_BESS', 'K_i_BESS', 'soc_min_BESS', 'soc_max_BESS', 'S_n_BESS', 'E_kWh_BESS', 'A_loss_BESS', 'B_loss_BESS', 'C_loss_BESS', 'R_bat_BESS', 'S_n_GRID', 'F_n_GRID', 'X_v_GRID', 'R_v_GRID', 'K_delta_GRID', 'K_alpha_GRID', 'K_rocov_GRID', 'I_sc_LV0101', 'I_mp_LV0101', 'V_mp_LV0101', 'V_oc_LV0101', 'N_pv_s_LV0101', 'N_pv_p_LV0101', 'K_vt_LV0101', 'K_it_LV0101', 'v_lvrt_LV0101', 'T_lp1p_LV0101', 'T_lp2p_LV0101', 'T_lp1q_LV0101', 'T_lp2q_LV0101', 'PRampUp_LV0101', 'PRampDown_LV0101', 'QRampUp_LV0101', 'QRampDown_LV0101', 'S_n_LV0101', 'F_n_LV0101', 'U_n_LV0101', 'X_s_LV0101', 'R_s_LV0101', 'I_sc_LV0102', 'I_mp_LV0102', 'V_mp_LV0102', 'V_oc_LV0102', 'N_pv_s_LV0102', 'N_pv_p_LV0102', 'K_vt_LV0102', 'K_it_LV0102', 'v_lvrt_LV0102', 'T_lp1p_LV0102', 'T_lp2p_LV0102', 'T_lp1q_LV0102', 'T_lp2q_LV0102', 'PRampUp_LV0102', 'PRampDown_LV0102', 'QRampUp_LV0102', 'QRampDown_LV0102', 'S_n_LV0102', 'F_n_LV0102', 'U_n_LV0102', 'X_s_LV0102', 'R_s_LV0102', 'I_sc_LV0103', 'I_mp_LV0103', 'V_mp_LV0103', 'V_oc_LV0103', 'N_pv_s_LV0103', 'N_pv_p_LV0103', 'K_vt_LV0103', 'K_it_LV0103', 'v_lvrt_LV0103', 'T_lp1p_LV0103', 'T_lp2p_LV0103', 'T_lp1q_LV0103', 'T_lp2q_LV0103', 'PRampUp_LV0103', 'PRampDown_LV0103', 'QRampUp_LV0103', 'QRampDown_LV0103', 'S_n_LV0103', 'F_n_LV0103', 'U_n_LV0103', 'X_s_LV0103', 'R_s_LV0103', 'I_sc_LV0104', 'I_mp_LV0104', 'V_mp_LV0104', 'V_oc_LV0104', 'N_pv_s_LV0104', 'N_pv_p_LV0104', 'K_vt_LV0104', 'K_it_LV0104', 'v_lvrt_LV0104', 'T_lp1p_LV0104', 'T_lp2p_LV0104', 'T_lp1q_LV0104', 'T_lp2q_LV0104', 'PRampUp_LV0104', 'PRampDown_LV0104', 'QRampUp_LV0104', 'QRampDown_LV0104', 'S_n_LV0104', 'F_n_LV0104', 'U_n_LV0104', 'X_s_LV0104', 'R_s_LV0104', 'I_sc_LV0105', 'I_mp_LV0105', 'V_mp_LV0105', 'V_oc_LV0105', 'N_pv_s_LV0105', 'N_pv_p_LV0105', 'K_vt_LV0105', 'K_it_LV0105', 'v_lvrt_LV0105', 'T_lp1p_LV0105', 'T_lp2p_LV0105', 'T_lp1q_LV0105', 'T_lp2q_LV0105', 'PRampUp_LV0105', 'PRampDown_LV0105', 'QRampUp_LV0105', 'QRampDown_LV0105', 'S_n_LV0105', 'F_n_LV0105', 'U_n_LV0105', 'X_s_LV0105', 'R_s_LV0105', 'I_sc_LV0201', 'I_mp_LV0201', 'V_mp_LV0201', 'V_oc_LV0201', 'N_pv_s_LV0201', 'N_pv_p_LV0201', 'K_vt_LV0201', 'K_it_LV0201', 'v_lvrt_LV0201', 'T_lp1p_LV0201', 'T_lp2p_LV0201', 'T_lp1q_LV0201', 'T_lp2q_LV0201', 'PRampUp_LV0201', 'PRampDown_LV0201', 'QRampUp_LV0201', 'QRampDown_LV0201', 'S_n_LV0201', 'F_n_LV0201', 'U_n_LV0201', 'X_s_LV0201', 'R_s_LV0201', 'I_sc_LV0202', 'I_mp_LV0202', 'V_mp_LV0202', 'V_oc_LV0202', 'N_pv_s_LV0202', 'N_pv_p_LV0202', 'K_vt_LV0202', 'K_it_LV0202', 'v_lvrt_LV0202', 'T_lp1p_LV0202', 'T_lp2p_LV0202', 'T_lp1q_LV0202', 'T_lp2q_LV0202', 'PRampUp_LV0202', 'PRampDown_LV0202', 'QRampUp_LV0202', 'QRampDown_LV0202', 'S_n_LV0202', 'F_n_LV0202', 'U_n_LV0202', 'X_s_LV0202', 'R_s_LV0202', 'I_sc_LV0203', 'I_mp_LV0203', 'V_mp_LV0203', 'V_oc_LV0203', 'N_pv_s_LV0203', 'N_pv_p_LV0203', 'K_vt_LV0203', 'K_it_LV0203', 'v_lvrt_LV0203', 'T_lp1p_LV0203', 'T_lp2p_LV0203', 'T_lp1q_LV0203', 'T_lp2q_LV0203', 'PRampUp_LV0203', 'PRampDown_LV0203', 'QRampUp_LV0203', 'QRampDown_LV0203', 'S_n_LV0203', 'F_n_LV0203', 'U_n_LV0203', 'X_s_LV0203', 'R_s_LV0203', 'I_sc_LV0204', 'I_mp_LV0204', 'V_mp_LV0204', 'V_oc_LV0204', 'N_pv_s_LV0204', 'N_pv_p_LV0204', 'K_vt_LV0204', 'K_it_LV0204', 'v_lvrt_LV0204', 'T_lp1p_LV0204', 'T_lp2p_LV0204', 'T_lp1q_LV0204', 'T_lp2q_LV0204', 'PRampUp_LV0204', 'PRampDown_LV0204', 'QRampUp_LV0204', 'QRampDown_LV0204', 'S_n_LV0204', 'F_n_LV0204', 'U_n_LV0204', 'X_s_LV0204', 'R_s_LV0204', 'I_sc_LV0205', 'I_mp_LV0205', 'V_mp_LV0205', 'V_oc_LV0205', 'N_pv_s_LV0205', 'N_pv_p_LV0205', 'K_vt_LV0205', 'K_it_LV0205', 'v_lvrt_LV0205', 'T_lp1p_LV0205', 'T_lp2p_LV0205', 'T_lp1q_LV0205', 'T_lp2q_LV0205', 'PRampUp_LV0205', 'PRampDown_LV0205', 'QRampUp_LV0205', 'QRampDown_LV0205', 'S_n_LV0205', 'F_n_LV0205', 'U_n_LV0205', 'X_s_LV0205', 'R_s_LV0205', 'I_sc_LV0301', 'I_mp_LV0301', 'V_mp_LV0301', 'V_oc_LV0301', 'N_pv_s_LV0301', 'N_pv_p_LV0301', 'K_vt_LV0301', 'K_it_LV0301', 'v_lvrt_LV0301', 'T_lp1p_LV0301', 'T_lp2p_LV0301', 'T_lp1q_LV0301', 'T_lp2q_LV0301', 'PRampUp_LV0301', 'PRampDown_LV0301', 'QRampUp_LV0301', 'QRampDown_LV0301', 'S_n_LV0301', 'F_n_LV0301', 'U_n_LV0301', 'X_s_LV0301', 'R_s_LV0301', 'I_sc_LV0302', 'I_mp_LV0302', 'V_mp_LV0302', 'V_oc_LV0302', 'N_pv_s_LV0302', 'N_pv_p_LV0302', 'K_vt_LV0302', 'K_it_LV0302', 'v_lvrt_LV0302', 'T_lp1p_LV0302', 'T_lp2p_LV0302', 'T_lp1q_LV0302', 'T_lp2q_LV0302', 'PRampUp_LV0302', 'PRampDown_LV0302', 'QRampUp_LV0302', 'QRampDown_LV0302', 'S_n_LV0302', 'F_n_LV0302', 'U_n_LV0302', 'X_s_LV0302', 'R_s_LV0302', 'I_sc_LV0303', 'I_mp_LV0303', 'V_mp_LV0303', 'V_oc_LV0303', 'N_pv_s_LV0303', 'N_pv_p_LV0303', 'K_vt_LV0303', 'K_it_LV0303', 'v_lvrt_LV0303', 'T_lp1p_LV0303', 'T_lp2p_LV0303', 'T_lp1q_LV0303', 'T_lp2q_LV0303', 'PRampUp_LV0303', 'PRampDown_LV0303', 'QRampUp_LV0303', 'QRampDown_LV0303', 'S_n_LV0303', 'F_n_LV0303', 'U_n_LV0303', 'X_s_LV0303', 'R_s_LV0303', 'I_sc_LV0304', 'I_mp_LV0304', 'V_mp_LV0304', 'V_oc_LV0304', 'N_pv_s_LV0304', 'N_pv_p_LV0304', 'K_vt_LV0304', 'K_it_LV0304', 'v_lvrt_LV0304', 'T_lp1p_LV0304', 'T_lp2p_LV0304', 'T_lp1q_LV0304', 'T_lp2q_LV0304', 'PRampUp_LV0304', 'PRampDown_LV0304', 'QRampUp_LV0304', 'QRampDown_LV0304', 'S_n_LV0304', 'F_n_LV0304', 'U_n_LV0304', 'X_s_LV0304', 'R_s_LV0304', 'I_sc_LV0305', 'I_mp_LV0305', 'V_mp_LV0305', 'V_oc_LV0305', 'N_pv_s_LV0305', 'N_pv_p_LV0305', 'K_vt_LV0305', 'K_it_LV0305', 'v_lvrt_LV0305', 'T_lp1p_LV0305', 'T_lp2p_LV0305', 'T_lp1q_LV0305', 'T_lp2q_LV0305', 'PRampUp_LV0305', 'PRampDown_LV0305', 'QRampUp_LV0305', 'QRampDown_LV0305', 'S_n_LV0305', 'F_n_LV0305', 'U_n_LV0305', 'X_s_LV0305', 'R_s_LV0305', 'I_sc_LV0401', 'I_mp_LV0401', 'V_mp_LV0401', 'V_oc_LV0401', 'N_pv_s_LV0401', 'N_pv_p_LV0401', 'K_vt_LV0401', 'K_it_LV0401', 'v_lvrt_LV0401', 'T_lp1p_LV0401', 'T_lp2p_LV0401', 'T_lp1q_LV0401', 'T_lp2q_LV0401', 'PRampUp_LV0401', 'PRampDown_LV0401', 'QRampUp_LV0401', 'QRampDown_LV0401', 'S_n_LV0401', 'F_n_LV0401', 'U_n_LV0401', 'X_s_LV0401', 'R_s_LV0401', 'I_sc_LV0402', 'I_mp_LV0402', 'V_mp_LV0402', 'V_oc_LV0402', 'N_pv_s_LV0402', 'N_pv_p_LV0402', 'K_vt_LV0402', 'K_it_LV0402', 'v_lvrt_LV0402', 'T_lp1p_LV0402', 'T_lp2p_LV0402', 'T_lp1q_LV0402', 'T_lp2q_LV0402', 'PRampUp_LV0402', 'PRampDown_LV0402', 'QRampUp_LV0402', 'QRampDown_LV0402', 'S_n_LV0402', 'F_n_LV0402', 'U_n_LV0402', 'X_s_LV0402', 'R_s_LV0402', 'I_sc_LV0403', 'I_mp_LV0403', 'V_mp_LV0403', 'V_oc_LV0403', 'N_pv_s_LV0403', 'N_pv_p_LV0403', 'K_vt_LV0403', 'K_it_LV0403', 'v_lvrt_LV0403', 'T_lp1p_LV0403', 'T_lp2p_LV0403', 'T_lp1q_LV0403', 'T_lp2q_LV0403', 'PRampUp_LV0403', 'PRampDown_LV0403', 'QRampUp_LV0403', 'QRampDown_LV0403', 'S_n_LV0403', 'F_n_LV0403', 'U_n_LV0403', 'X_s_LV0403', 'R_s_LV0403', 'I_sc_LV0404', 'I_mp_LV0404', 'V_mp_LV0404', 'V_oc_LV0404', 'N_pv_s_LV0404', 'N_pv_p_LV0404', 'K_vt_LV0404', 'K_it_LV0404', 'v_lvrt_LV0404', 'T_lp1p_LV0404', 'T_lp2p_LV0404', 'T_lp1q_LV0404', 'T_lp2q_LV0404', 'PRampUp_LV0404', 'PRampDown_LV0404', 'QRampUp_LV0404', 'QRampDown_LV0404', 'S_n_LV0404', 'F_n_LV0404', 'U_n_LV0404', 'X_s_LV0404', 'R_s_LV0404', 'I_sc_LV0405', 'I_mp_LV0405', 'V_mp_LV0405', 'V_oc_LV0405', 'N_pv_s_LV0405', 'N_pv_p_LV0405', 'K_vt_LV0405', 'K_it_LV0405', 'v_lvrt_LV0405', 'T_lp1p_LV0405', 'T_lp2p_LV0405', 'T_lp1q_LV0405', 'T_lp2q_LV0405', 'PRampUp_LV0405', 'PRampDown_LV0405', 'QRampUp_LV0405', 'QRampDown_LV0405', 'S_n_LV0405', 'F_n_LV0405', 'U_n_LV0405', 'X_s_LV0405', 'R_s_LV0405', 'K_p_agc', 'K_i_agc', 'K_xif'] 
        self.params_values_list  = [100000000.0, 0.0, -2.12, 0.0, 0.0, -0.23999999999999996, 0.0, 0.04615384615384615, -0.23076923076923075, 0.0, 3.0, -3.0, 0.0006, 0.04615384615384615, -0.23076923076923075, 0.0, 2.4, -2.4, 0.00048, 0.04615384615384615, -0.23076923076923075, 0.0, 1.7999999999999998, -1.7999999999999998, 0.0003599999999999999, 0.04615384615384615, -0.23076923076923075, 0.0, 1.2, -1.2, 0.00024, 0.04615384615384615, -0.23076923076923075, 0.0, 0.6, -0.6, 0.00012, 0.04615384615384615, -0.23076923076923075, 0.0, 3.0, -3.0, 0.0006, 0.04615384615384615, -0.23076923076923075, 0.0, 2.4, -2.4, 0.00048, 0.04615384615384615, -0.23076923076923075, 0.0, 1.7999999999999998, -1.7999999999999998, 0.0003599999999999999, 0.04615384615384615, -0.23076923076923075, 0.0, 1.2, -1.2, 0.00024, 0.04615384615384615, -0.23076923076923075, 0.0, 0.6, -0.6, 0.00012, 0.04615384615384615, -0.23076923076923075, 0.0, 3.0, -3.0, 0.0006, 0.04615384615384615, -0.23076923076923075, 0.0, 2.4, -2.4, 0.00048, 0.04615384615384615, -0.23076923076923075, 0.0, 1.7999999999999998, -1.7999999999999998, 0.0003599999999999999, 0.04615384615384615, -0.23076923076923075, 0.0, 1.2, -1.2, 0.00024, 0.04615384615384615, -0.23076923076923075, 0.0, 0.6, -0.6, 0.00012, 0.04615384615384615, -0.23076923076923075, 0.0, 3.0, -3.0, 0.0006, 0.04615384615384615, -0.23076923076923075, 0.0, 2.4, -2.4, 0.00048, 0.04615384615384615, -0.23076923076923075, 0.0, 1.7999999999999998, -1.7999999999999998, 0.0003599999999999999, 0.04615384615384615, -0.23076923076923075, 0.0, 1.2, -1.2, 0.00024, 0.04615384615384615, -0.23076923076923075, 0.0, 0.6, -0.6, 0.00012, 0.0, -4.0, 1.0, 0.0, 20000.0, 132000, 132000, 690.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 1e-06, 1e-06, 0.0, 1.0, 1000000.0, 250, 0.0001, 0.0, 0.0001, 0.0, 1000000000.0, 50.0, 0.001, 0.0, 0.001, 1e-06, 1e-06, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 0.0, 0.0, 0.01] 
        self.inputs_ini_list = ['P_POI_MV', 'Q_POI_MV', 'P_POI', 'Q_POI', 'P_GRID', 'Q_GRID', 'P_BESS', 'Q_BESS', 'P_LV0101', 'Q_LV0101', 'P_MV0101', 'Q_MV0101', 'P_LV0102', 'Q_LV0102', 'P_MV0102', 'Q_MV0102', 'P_LV0103', 'Q_LV0103', 'P_MV0103', 'Q_MV0103', 'P_LV0104', 'Q_LV0104', 'P_MV0104', 'Q_MV0104', 'P_LV0105', 'Q_LV0105', 'P_MV0105', 'Q_MV0105', 'P_LV0201', 'Q_LV0201', 'P_MV0201', 'Q_MV0201', 'P_LV0202', 'Q_LV0202', 'P_MV0202', 'Q_MV0202', 'P_LV0203', 'Q_LV0203', 'P_MV0203', 'Q_MV0203', 'P_LV0204', 'Q_LV0204', 'P_MV0204', 'Q_MV0204', 'P_LV0205', 'Q_LV0205', 'P_MV0205', 'Q_MV0205', 'P_LV0301', 'Q_LV0301', 'P_MV0301', 'Q_MV0301', 'P_LV0302', 'Q_LV0302', 'P_MV0302', 'Q_MV0302', 'P_LV0303', 'Q_LV0303', 'P_MV0303', 'Q_MV0303', 'P_LV0304', 'Q_LV0304', 'P_MV0304', 'Q_MV0304', 'P_LV0305', 'Q_LV0305', 'P_MV0305', 'Q_MV0305', 'P_LV0401', 'Q_LV0401', 'P_MV0401', 'Q_MV0401', 'P_LV0402', 'Q_LV0402', 'P_MV0402', 'Q_MV0402', 'P_LV0403', 'Q_LV0403', 'P_MV0403', 'Q_MV0403', 'P_LV0404', 'Q_LV0404', 'P_MV0404', 'Q_MV0404', 'P_LV0405', 'Q_LV0405', 'P_MV0405', 'Q_MV0405', 'p_s_ref_BESS', 'q_s_ref_BESS', 'soc_ref_BESS', 'alpha_GRID', 'v_ref_GRID', 'omega_ref_GRID', 'delta_ref_GRID', 'phi_GRID', 'rocov_GRID', 'irrad_LV0101', 'temp_deg_LV0101', 'lvrt_ext_LV0101', 'ramp_enable_LV0101', 'p_s_ppc_LV0101', 'q_s_ppc_LV0101', 'i_sa_ref_LV0101', 'i_sr_ref_LV0101', 'irrad_LV0102', 'temp_deg_LV0102', 'lvrt_ext_LV0102', 'ramp_enable_LV0102', 'p_s_ppc_LV0102', 'q_s_ppc_LV0102', 'i_sa_ref_LV0102', 'i_sr_ref_LV0102', 'irrad_LV0103', 'temp_deg_LV0103', 'lvrt_ext_LV0103', 'ramp_enable_LV0103', 'p_s_ppc_LV0103', 'q_s_ppc_LV0103', 'i_sa_ref_LV0103', 'i_sr_ref_LV0103', 'irrad_LV0104', 'temp_deg_LV0104', 'lvrt_ext_LV0104', 'ramp_enable_LV0104', 'p_s_ppc_LV0104', 'q_s_ppc_LV0104', 'i_sa_ref_LV0104', 'i_sr_ref_LV0104', 'irrad_LV0105', 'temp_deg_LV0105', 'lvrt_ext_LV0105', 'ramp_enable_LV0105', 'p_s_ppc_LV0105', 'q_s_ppc_LV0105', 'i_sa_ref_LV0105', 'i_sr_ref_LV0105', 'irrad_LV0201', 'temp_deg_LV0201', 'lvrt_ext_LV0201', 'ramp_enable_LV0201', 'p_s_ppc_LV0201', 'q_s_ppc_LV0201', 'i_sa_ref_LV0201', 'i_sr_ref_LV0201', 'irrad_LV0202', 'temp_deg_LV0202', 'lvrt_ext_LV0202', 'ramp_enable_LV0202', 'p_s_ppc_LV0202', 'q_s_ppc_LV0202', 'i_sa_ref_LV0202', 'i_sr_ref_LV0202', 'irrad_LV0203', 'temp_deg_LV0203', 'lvrt_ext_LV0203', 'ramp_enable_LV0203', 'p_s_ppc_LV0203', 'q_s_ppc_LV0203', 'i_sa_ref_LV0203', 'i_sr_ref_LV0203', 'irrad_LV0204', 'temp_deg_LV0204', 'lvrt_ext_LV0204', 'ramp_enable_LV0204', 'p_s_ppc_LV0204', 'q_s_ppc_LV0204', 'i_sa_ref_LV0204', 'i_sr_ref_LV0204', 'irrad_LV0205', 'temp_deg_LV0205', 'lvrt_ext_LV0205', 'ramp_enable_LV0205', 'p_s_ppc_LV0205', 'q_s_ppc_LV0205', 'i_sa_ref_LV0205', 'i_sr_ref_LV0205', 'irrad_LV0301', 'temp_deg_LV0301', 'lvrt_ext_LV0301', 'ramp_enable_LV0301', 'p_s_ppc_LV0301', 'q_s_ppc_LV0301', 'i_sa_ref_LV0301', 'i_sr_ref_LV0301', 'irrad_LV0302', 'temp_deg_LV0302', 'lvrt_ext_LV0302', 'ramp_enable_LV0302', 'p_s_ppc_LV0302', 'q_s_ppc_LV0302', 'i_sa_ref_LV0302', 'i_sr_ref_LV0302', 'irrad_LV0303', 'temp_deg_LV0303', 'lvrt_ext_LV0303', 'ramp_enable_LV0303', 'p_s_ppc_LV0303', 'q_s_ppc_LV0303', 'i_sa_ref_LV0303', 'i_sr_ref_LV0303', 'irrad_LV0304', 'temp_deg_LV0304', 'lvrt_ext_LV0304', 'ramp_enable_LV0304', 'p_s_ppc_LV0304', 'q_s_ppc_LV0304', 'i_sa_ref_LV0304', 'i_sr_ref_LV0304', 'irrad_LV0305', 'temp_deg_LV0305', 'lvrt_ext_LV0305', 'ramp_enable_LV0305', 'p_s_ppc_LV0305', 'q_s_ppc_LV0305', 'i_sa_ref_LV0305', 'i_sr_ref_LV0305', 'irrad_LV0401', 'temp_deg_LV0401', 'lvrt_ext_LV0401', 'ramp_enable_LV0401', 'p_s_ppc_LV0401', 'q_s_ppc_LV0401', 'i_sa_ref_LV0401', 'i_sr_ref_LV0401', 'irrad_LV0402', 'temp_deg_LV0402', 'lvrt_ext_LV0402', 'ramp_enable_LV0402', 'p_s_ppc_LV0402', 'q_s_ppc_LV0402', 'i_sa_ref_LV0402', 'i_sr_ref_LV0402', 'irrad_LV0403', 'temp_deg_LV0403', 'lvrt_ext_LV0403', 'ramp_enable_LV0403', 'p_s_ppc_LV0403', 'q_s_ppc_LV0403', 'i_sa_ref_LV0403', 'i_sr_ref_LV0403', 'irrad_LV0404', 'temp_deg_LV0404', 'lvrt_ext_LV0404', 'ramp_enable_LV0404', 'p_s_ppc_LV0404', 'q_s_ppc_LV0404', 'i_sa_ref_LV0404', 'i_sr_ref_LV0404', 'irrad_LV0405', 'temp_deg_LV0405', 'lvrt_ext_LV0405', 'ramp_enable_LV0405', 'p_s_ppc_LV0405', 'q_s_ppc_LV0405', 'i_sa_ref_LV0405', 'i_sr_ref_LV0405'] 
        self.inputs_ini_values_list  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0, 1.0, 1.0, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0] 
        self.inputs_run_list = ['P_POI_MV', 'Q_POI_MV', 'P_POI', 'Q_POI', 'P_GRID', 'Q_GRID', 'P_BESS', 'Q_BESS', 'P_LV0101', 'Q_LV0101', 'P_MV0101', 'Q_MV0101', 'P_LV0102', 'Q_LV0102', 'P_MV0102', 'Q_MV0102', 'P_LV0103', 'Q_LV0103', 'P_MV0103', 'Q_MV0103', 'P_LV0104', 'Q_LV0104', 'P_MV0104', 'Q_MV0104', 'P_LV0105', 'Q_LV0105', 'P_MV0105', 'Q_MV0105', 'P_LV0201', 'Q_LV0201', 'P_MV0201', 'Q_MV0201', 'P_LV0202', 'Q_LV0202', 'P_MV0202', 'Q_MV0202', 'P_LV0203', 'Q_LV0203', 'P_MV0203', 'Q_MV0203', 'P_LV0204', 'Q_LV0204', 'P_MV0204', 'Q_MV0204', 'P_LV0205', 'Q_LV0205', 'P_MV0205', 'Q_MV0205', 'P_LV0301', 'Q_LV0301', 'P_MV0301', 'Q_MV0301', 'P_LV0302', 'Q_LV0302', 'P_MV0302', 'Q_MV0302', 'P_LV0303', 'Q_LV0303', 'P_MV0303', 'Q_MV0303', 'P_LV0304', 'Q_LV0304', 'P_MV0304', 'Q_MV0304', 'P_LV0305', 'Q_LV0305', 'P_MV0305', 'Q_MV0305', 'P_LV0401', 'Q_LV0401', 'P_MV0401', 'Q_MV0401', 'P_LV0402', 'Q_LV0402', 'P_MV0402', 'Q_MV0402', 'P_LV0403', 'Q_LV0403', 'P_MV0403', 'Q_MV0403', 'P_LV0404', 'Q_LV0404', 'P_MV0404', 'Q_MV0404', 'P_LV0405', 'Q_LV0405', 'P_MV0405', 'Q_MV0405', 'p_s_ref_BESS', 'q_s_ref_BESS', 'soc_ref_BESS', 'alpha_GRID', 'v_ref_GRID', 'omega_ref_GRID', 'delta_ref_GRID', 'phi_GRID', 'rocov_GRID', 'irrad_LV0101', 'temp_deg_LV0101', 'lvrt_ext_LV0101', 'ramp_enable_LV0101', 'p_s_ppc_LV0101', 'q_s_ppc_LV0101', 'i_sa_ref_LV0101', 'i_sr_ref_LV0101', 'irrad_LV0102', 'temp_deg_LV0102', 'lvrt_ext_LV0102', 'ramp_enable_LV0102', 'p_s_ppc_LV0102', 'q_s_ppc_LV0102', 'i_sa_ref_LV0102', 'i_sr_ref_LV0102', 'irrad_LV0103', 'temp_deg_LV0103', 'lvrt_ext_LV0103', 'ramp_enable_LV0103', 'p_s_ppc_LV0103', 'q_s_ppc_LV0103', 'i_sa_ref_LV0103', 'i_sr_ref_LV0103', 'irrad_LV0104', 'temp_deg_LV0104', 'lvrt_ext_LV0104', 'ramp_enable_LV0104', 'p_s_ppc_LV0104', 'q_s_ppc_LV0104', 'i_sa_ref_LV0104', 'i_sr_ref_LV0104', 'irrad_LV0105', 'temp_deg_LV0105', 'lvrt_ext_LV0105', 'ramp_enable_LV0105', 'p_s_ppc_LV0105', 'q_s_ppc_LV0105', 'i_sa_ref_LV0105', 'i_sr_ref_LV0105', 'irrad_LV0201', 'temp_deg_LV0201', 'lvrt_ext_LV0201', 'ramp_enable_LV0201', 'p_s_ppc_LV0201', 'q_s_ppc_LV0201', 'i_sa_ref_LV0201', 'i_sr_ref_LV0201', 'irrad_LV0202', 'temp_deg_LV0202', 'lvrt_ext_LV0202', 'ramp_enable_LV0202', 'p_s_ppc_LV0202', 'q_s_ppc_LV0202', 'i_sa_ref_LV0202', 'i_sr_ref_LV0202', 'irrad_LV0203', 'temp_deg_LV0203', 'lvrt_ext_LV0203', 'ramp_enable_LV0203', 'p_s_ppc_LV0203', 'q_s_ppc_LV0203', 'i_sa_ref_LV0203', 'i_sr_ref_LV0203', 'irrad_LV0204', 'temp_deg_LV0204', 'lvrt_ext_LV0204', 'ramp_enable_LV0204', 'p_s_ppc_LV0204', 'q_s_ppc_LV0204', 'i_sa_ref_LV0204', 'i_sr_ref_LV0204', 'irrad_LV0205', 'temp_deg_LV0205', 'lvrt_ext_LV0205', 'ramp_enable_LV0205', 'p_s_ppc_LV0205', 'q_s_ppc_LV0205', 'i_sa_ref_LV0205', 'i_sr_ref_LV0205', 'irrad_LV0301', 'temp_deg_LV0301', 'lvrt_ext_LV0301', 'ramp_enable_LV0301', 'p_s_ppc_LV0301', 'q_s_ppc_LV0301', 'i_sa_ref_LV0301', 'i_sr_ref_LV0301', 'irrad_LV0302', 'temp_deg_LV0302', 'lvrt_ext_LV0302', 'ramp_enable_LV0302', 'p_s_ppc_LV0302', 'q_s_ppc_LV0302', 'i_sa_ref_LV0302', 'i_sr_ref_LV0302', 'irrad_LV0303', 'temp_deg_LV0303', 'lvrt_ext_LV0303', 'ramp_enable_LV0303', 'p_s_ppc_LV0303', 'q_s_ppc_LV0303', 'i_sa_ref_LV0303', 'i_sr_ref_LV0303', 'irrad_LV0304', 'temp_deg_LV0304', 'lvrt_ext_LV0304', 'ramp_enable_LV0304', 'p_s_ppc_LV0304', 'q_s_ppc_LV0304', 'i_sa_ref_LV0304', 'i_sr_ref_LV0304', 'irrad_LV0305', 'temp_deg_LV0305', 'lvrt_ext_LV0305', 'ramp_enable_LV0305', 'p_s_ppc_LV0305', 'q_s_ppc_LV0305', 'i_sa_ref_LV0305', 'i_sr_ref_LV0305', 'irrad_LV0401', 'temp_deg_LV0401', 'lvrt_ext_LV0401', 'ramp_enable_LV0401', 'p_s_ppc_LV0401', 'q_s_ppc_LV0401', 'i_sa_ref_LV0401', 'i_sr_ref_LV0401', 'irrad_LV0402', 'temp_deg_LV0402', 'lvrt_ext_LV0402', 'ramp_enable_LV0402', 'p_s_ppc_LV0402', 'q_s_ppc_LV0402', 'i_sa_ref_LV0402', 'i_sr_ref_LV0402', 'irrad_LV0403', 'temp_deg_LV0403', 'lvrt_ext_LV0403', 'ramp_enable_LV0403', 'p_s_ppc_LV0403', 'q_s_ppc_LV0403', 'i_sa_ref_LV0403', 'i_sr_ref_LV0403', 'irrad_LV0404', 'temp_deg_LV0404', 'lvrt_ext_LV0404', 'ramp_enable_LV0404', 'p_s_ppc_LV0404', 'q_s_ppc_LV0404', 'i_sa_ref_LV0404', 'i_sr_ref_LV0404', 'irrad_LV0405', 'temp_deg_LV0405', 'lvrt_ext_LV0405', 'ramp_enable_LV0405', 'p_s_ppc_LV0405', 'q_s_ppc_LV0405', 'i_sa_ref_LV0405', 'i_sr_ref_LV0405'] 
        self.inputs_run_values_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0, 1.0, 1.0, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0] 
        self.outputs_list = ['V_POI_MV', 'V_POI', 'V_GRID', 'V_BESS', 'V_LV0101', 'V_MV0101', 'V_LV0102', 'V_MV0102', 'V_LV0103', 'V_MV0103', 'V_LV0104', 'V_MV0104', 'V_LV0105', 'V_MV0105', 'V_LV0201', 'V_MV0201', 'V_LV0202', 'V_MV0202', 'V_LV0203', 'V_MV0203', 'V_LV0204', 'V_MV0204', 'V_LV0205', 'V_MV0205', 'V_LV0301', 'V_MV0301', 'V_LV0302', 'V_MV0302', 'V_LV0303', 'V_MV0303', 'V_LV0304', 'V_MV0304', 'V_LV0305', 'V_MV0305', 'V_LV0401', 'V_MV0401', 'V_LV0402', 'V_MV0402', 'V_LV0403', 'V_MV0403', 'V_LV0404', 'V_MV0404', 'V_LV0405', 'V_MV0405', 'p_line_POI_GRID', 'q_line_POI_GRID', 'p_line_GRID_POI', 'q_line_GRID_POI', 'p_line_BESS_POI_MV', 'q_line_BESS_POI_MV', 'p_line_POI_MV_BESS', 'q_line_POI_MV_BESS', 'p_line_MV0101_POI_MV', 'q_line_MV0101_POI_MV', 'p_line_POI_MV_MV0101', 'q_line_POI_MV_MV0101', 'p_line_MV0201_POI_MV', 'q_line_MV0201_POI_MV', 'p_line_POI_MV_MV0201', 'q_line_POI_MV_MV0201', 'p_line_MV0301_POI_MV', 'q_line_MV0301_POI_MV', 'p_line_POI_MV_MV0301', 'q_line_POI_MV_MV0301', 'p_line_MV0401_POI_MV', 'q_line_MV0401_POI_MV', 'p_line_POI_MV_MV0401', 'q_line_POI_MV_MV0401', 'p_loss_BESS', 'i_s_BESS', 'e_BESS', 'i_dc_BESS', 'p_s_BESS', 'q_s_BESS', 'alpha_GRID', 'Dv_GRID', 'm_ref_LV0101', 'v_sd_LV0101', 'v_sq_LV0101', 'lvrt_LV0101', 'm_ref_LV0102', 'v_sd_LV0102', 'v_sq_LV0102', 'lvrt_LV0102', 'm_ref_LV0103', 'v_sd_LV0103', 'v_sq_LV0103', 'lvrt_LV0103', 'm_ref_LV0104', 'v_sd_LV0104', 'v_sq_LV0104', 'lvrt_LV0104', 'm_ref_LV0105', 'v_sd_LV0105', 'v_sq_LV0105', 'lvrt_LV0105', 'm_ref_LV0201', 'v_sd_LV0201', 'v_sq_LV0201', 'lvrt_LV0201', 'm_ref_LV0202', 'v_sd_LV0202', 'v_sq_LV0202', 'lvrt_LV0202', 'm_ref_LV0203', 'v_sd_LV0203', 'v_sq_LV0203', 'lvrt_LV0203', 'm_ref_LV0204', 'v_sd_LV0204', 'v_sq_LV0204', 'lvrt_LV0204', 'm_ref_LV0205', 'v_sd_LV0205', 'v_sq_LV0205', 'lvrt_LV0205', 'm_ref_LV0301', 'v_sd_LV0301', 'v_sq_LV0301', 'lvrt_LV0301', 'm_ref_LV0302', 'v_sd_LV0302', 'v_sq_LV0302', 'lvrt_LV0302', 'm_ref_LV0303', 'v_sd_LV0303', 'v_sq_LV0303', 'lvrt_LV0303', 'm_ref_LV0304', 'v_sd_LV0304', 'v_sq_LV0304', 'lvrt_LV0304', 'm_ref_LV0305', 'v_sd_LV0305', 'v_sq_LV0305', 'lvrt_LV0305', 'm_ref_LV0401', 'v_sd_LV0401', 'v_sq_LV0401', 'lvrt_LV0401', 'm_ref_LV0402', 'v_sd_LV0402', 'v_sq_LV0402', 'lvrt_LV0402', 'm_ref_LV0403', 'v_sd_LV0403', 'v_sq_LV0403', 'lvrt_LV0403', 'm_ref_LV0404', 'v_sd_LV0404', 'v_sq_LV0404', 'lvrt_LV0404', 'm_ref_LV0405', 'v_sd_LV0405', 'v_sq_LV0405', 'lvrt_LV0405'] 
        self.x_list = ['soc_BESS', 'xi_soc_BESS', 'delta_GRID', 'Domega_GRID', 'Dv_GRID', 'x_p_lp1_LV0101', 'x_p_lp2_LV0101', 'x_q_lp1_LV0101', 'x_q_lp2_LV0101', 'x_p_lp1_LV0102', 'x_p_lp2_LV0102', 'x_q_lp1_LV0102', 'x_q_lp2_LV0102', 'x_p_lp1_LV0103', 'x_p_lp2_LV0103', 'x_q_lp1_LV0103', 'x_q_lp2_LV0103', 'x_p_lp1_LV0104', 'x_p_lp2_LV0104', 'x_q_lp1_LV0104', 'x_q_lp2_LV0104', 'x_p_lp1_LV0105', 'x_p_lp2_LV0105', 'x_q_lp1_LV0105', 'x_q_lp2_LV0105', 'x_p_lp1_LV0201', 'x_p_lp2_LV0201', 'x_q_lp1_LV0201', 'x_q_lp2_LV0201', 'x_p_lp1_LV0202', 'x_p_lp2_LV0202', 'x_q_lp1_LV0202', 'x_q_lp2_LV0202', 'x_p_lp1_LV0203', 'x_p_lp2_LV0203', 'x_q_lp1_LV0203', 'x_q_lp2_LV0203', 'x_p_lp1_LV0204', 'x_p_lp2_LV0204', 'x_q_lp1_LV0204', 'x_q_lp2_LV0204', 'x_p_lp1_LV0205', 'x_p_lp2_LV0205', 'x_q_lp1_LV0205', 'x_q_lp2_LV0205', 'x_p_lp1_LV0301', 'x_p_lp2_LV0301', 'x_q_lp1_LV0301', 'x_q_lp2_LV0301', 'x_p_lp1_LV0302', 'x_p_lp2_LV0302', 'x_q_lp1_LV0302', 'x_q_lp2_LV0302', 'x_p_lp1_LV0303', 'x_p_lp2_LV0303', 'x_q_lp1_LV0303', 'x_q_lp2_LV0303', 'x_p_lp1_LV0304', 'x_p_lp2_LV0304', 'x_q_lp1_LV0304', 'x_q_lp2_LV0304', 'x_p_lp1_LV0305', 'x_p_lp2_LV0305', 'x_q_lp1_LV0305', 'x_q_lp2_LV0305', 'x_p_lp1_LV0401', 'x_p_lp2_LV0401', 'x_q_lp1_LV0401', 'x_q_lp2_LV0401', 'x_p_lp1_LV0402', 'x_p_lp2_LV0402', 'x_q_lp1_LV0402', 'x_q_lp2_LV0402', 'x_p_lp1_LV0403', 'x_p_lp2_LV0403', 'x_q_lp1_LV0403', 'x_q_lp2_LV0403', 'x_p_lp1_LV0404', 'x_p_lp2_LV0404', 'x_q_lp1_LV0404', 'x_q_lp2_LV0404', 'x_p_lp1_LV0405', 'x_p_lp2_LV0405', 'x_q_lp1_LV0405', 'x_q_lp2_LV0405', 'xi_freq'] 
        self.y_run_list = ['V_POI_MV', 'theta_POI_MV', 'V_POI', 'theta_POI', 'V_GRID', 'theta_GRID', 'V_BESS', 'theta_BESS', 'V_LV0101', 'theta_LV0101', 'V_MV0101', 'theta_MV0101', 'V_LV0102', 'theta_LV0102', 'V_MV0102', 'theta_MV0102', 'V_LV0103', 'theta_LV0103', 'V_MV0103', 'theta_MV0103', 'V_LV0104', 'theta_LV0104', 'V_MV0104', 'theta_MV0104', 'V_LV0105', 'theta_LV0105', 'V_MV0105', 'theta_MV0105', 'V_LV0201', 'theta_LV0201', 'V_MV0201', 'theta_MV0201', 'V_LV0202', 'theta_LV0202', 'V_MV0202', 'theta_MV0202', 'V_LV0203', 'theta_LV0203', 'V_MV0203', 'theta_MV0203', 'V_LV0204', 'theta_LV0204', 'V_MV0204', 'theta_MV0204', 'V_LV0205', 'theta_LV0205', 'V_MV0205', 'theta_MV0205', 'V_LV0301', 'theta_LV0301', 'V_MV0301', 'theta_MV0301', 'V_LV0302', 'theta_LV0302', 'V_MV0302', 'theta_MV0302', 'V_LV0303', 'theta_LV0303', 'V_MV0303', 'theta_MV0303', 'V_LV0304', 'theta_LV0304', 'V_MV0304', 'theta_MV0304', 'V_LV0305', 'theta_LV0305', 'V_MV0305', 'theta_MV0305', 'V_LV0401', 'theta_LV0401', 'V_MV0401', 'theta_MV0401', 'V_LV0402', 'theta_LV0402', 'V_MV0402', 'theta_MV0402', 'V_LV0403', 'theta_LV0403', 'V_MV0403', 'theta_MV0403', 'V_LV0404', 'theta_LV0404', 'V_MV0404', 'theta_MV0404', 'V_LV0405', 'theta_LV0405', 'V_MV0405', 'theta_MV0405', 'p_dc_BESS', 'i_dc_BESS', 'v_dc_BESS', 'omega_GRID', 'i_d_GRID', 'i_q_GRID', 'p_s_GRID', 'q_s_GRID', 'v_dc_LV0101', 'i_sq_ref_LV0101', 'i_sd_ref_LV0101', 'i_sr_LV0101', 'i_si_LV0101', 'p_s_LV0101', 'q_s_LV0101', 'v_dc_LV0102', 'i_sq_ref_LV0102', 'i_sd_ref_LV0102', 'i_sr_LV0102', 'i_si_LV0102', 'p_s_LV0102', 'q_s_LV0102', 'v_dc_LV0103', 'i_sq_ref_LV0103', 'i_sd_ref_LV0103', 'i_sr_LV0103', 'i_si_LV0103', 'p_s_LV0103', 'q_s_LV0103', 'v_dc_LV0104', 'i_sq_ref_LV0104', 'i_sd_ref_LV0104', 'i_sr_LV0104', 'i_si_LV0104', 'p_s_LV0104', 'q_s_LV0104', 'v_dc_LV0105', 'i_sq_ref_LV0105', 'i_sd_ref_LV0105', 'i_sr_LV0105', 'i_si_LV0105', 'p_s_LV0105', 'q_s_LV0105', 'v_dc_LV0201', 'i_sq_ref_LV0201', 'i_sd_ref_LV0201', 'i_sr_LV0201', 'i_si_LV0201', 'p_s_LV0201', 'q_s_LV0201', 'v_dc_LV0202', 'i_sq_ref_LV0202', 'i_sd_ref_LV0202', 'i_sr_LV0202', 'i_si_LV0202', 'p_s_LV0202', 'q_s_LV0202', 'v_dc_LV0203', 'i_sq_ref_LV0203', 'i_sd_ref_LV0203', 'i_sr_LV0203', 'i_si_LV0203', 'p_s_LV0203', 'q_s_LV0203', 'v_dc_LV0204', 'i_sq_ref_LV0204', 'i_sd_ref_LV0204', 'i_sr_LV0204', 'i_si_LV0204', 'p_s_LV0204', 'q_s_LV0204', 'v_dc_LV0205', 'i_sq_ref_LV0205', 'i_sd_ref_LV0205', 'i_sr_LV0205', 'i_si_LV0205', 'p_s_LV0205', 'q_s_LV0205', 'v_dc_LV0301', 'i_sq_ref_LV0301', 'i_sd_ref_LV0301', 'i_sr_LV0301', 'i_si_LV0301', 'p_s_LV0301', 'q_s_LV0301', 'v_dc_LV0302', 'i_sq_ref_LV0302', 'i_sd_ref_LV0302', 'i_sr_LV0302', 'i_si_LV0302', 'p_s_LV0302', 'q_s_LV0302', 'v_dc_LV0303', 'i_sq_ref_LV0303', 'i_sd_ref_LV0303', 'i_sr_LV0303', 'i_si_LV0303', 'p_s_LV0303', 'q_s_LV0303', 'v_dc_LV0304', 'i_sq_ref_LV0304', 'i_sd_ref_LV0304', 'i_sr_LV0304', 'i_si_LV0304', 'p_s_LV0304', 'q_s_LV0304', 'v_dc_LV0305', 'i_sq_ref_LV0305', 'i_sd_ref_LV0305', 'i_sr_LV0305', 'i_si_LV0305', 'p_s_LV0305', 'q_s_LV0305', 'v_dc_LV0401', 'i_sq_ref_LV0401', 'i_sd_ref_LV0401', 'i_sr_LV0401', 'i_si_LV0401', 'p_s_LV0401', 'q_s_LV0401', 'v_dc_LV0402', 'i_sq_ref_LV0402', 'i_sd_ref_LV0402', 'i_sr_LV0402', 'i_si_LV0402', 'p_s_LV0402', 'q_s_LV0402', 'v_dc_LV0403', 'i_sq_ref_LV0403', 'i_sd_ref_LV0403', 'i_sr_LV0403', 'i_si_LV0403', 'p_s_LV0403', 'q_s_LV0403', 'v_dc_LV0404', 'i_sq_ref_LV0404', 'i_sd_ref_LV0404', 'i_sr_LV0404', 'i_si_LV0404', 'p_s_LV0404', 'q_s_LV0404', 'v_dc_LV0405', 'i_sq_ref_LV0405', 'i_sd_ref_LV0405', 'i_sr_LV0405', 'i_si_LV0405', 'p_s_LV0405', 'q_s_LV0405', 'omega_coi', 'p_agc'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['V_POI_MV', 'theta_POI_MV', 'V_POI', 'theta_POI', 'V_GRID', 'theta_GRID', 'V_BESS', 'theta_BESS', 'V_LV0101', 'theta_LV0101', 'V_MV0101', 'theta_MV0101', 'V_LV0102', 'theta_LV0102', 'V_MV0102', 'theta_MV0102', 'V_LV0103', 'theta_LV0103', 'V_MV0103', 'theta_MV0103', 'V_LV0104', 'theta_LV0104', 'V_MV0104', 'theta_MV0104', 'V_LV0105', 'theta_LV0105', 'V_MV0105', 'theta_MV0105', 'V_LV0201', 'theta_LV0201', 'V_MV0201', 'theta_MV0201', 'V_LV0202', 'theta_LV0202', 'V_MV0202', 'theta_MV0202', 'V_LV0203', 'theta_LV0203', 'V_MV0203', 'theta_MV0203', 'V_LV0204', 'theta_LV0204', 'V_MV0204', 'theta_MV0204', 'V_LV0205', 'theta_LV0205', 'V_MV0205', 'theta_MV0205', 'V_LV0301', 'theta_LV0301', 'V_MV0301', 'theta_MV0301', 'V_LV0302', 'theta_LV0302', 'V_MV0302', 'theta_MV0302', 'V_LV0303', 'theta_LV0303', 'V_MV0303', 'theta_MV0303', 'V_LV0304', 'theta_LV0304', 'V_MV0304', 'theta_MV0304', 'V_LV0305', 'theta_LV0305', 'V_MV0305', 'theta_MV0305', 'V_LV0401', 'theta_LV0401', 'V_MV0401', 'theta_MV0401', 'V_LV0402', 'theta_LV0402', 'V_MV0402', 'theta_MV0402', 'V_LV0403', 'theta_LV0403', 'V_MV0403', 'theta_MV0403', 'V_LV0404', 'theta_LV0404', 'V_MV0404', 'theta_MV0404', 'V_LV0405', 'theta_LV0405', 'V_MV0405', 'theta_MV0405', 'p_dc_BESS', 'i_dc_BESS', 'v_dc_BESS', 'omega_GRID', 'i_d_GRID', 'i_q_GRID', 'p_s_GRID', 'q_s_GRID', 'v_dc_LV0101', 'i_sq_ref_LV0101', 'i_sd_ref_LV0101', 'i_sr_LV0101', 'i_si_LV0101', 'p_s_LV0101', 'q_s_LV0101', 'v_dc_LV0102', 'i_sq_ref_LV0102', 'i_sd_ref_LV0102', 'i_sr_LV0102', 'i_si_LV0102', 'p_s_LV0102', 'q_s_LV0102', 'v_dc_LV0103', 'i_sq_ref_LV0103', 'i_sd_ref_LV0103', 'i_sr_LV0103', 'i_si_LV0103', 'p_s_LV0103', 'q_s_LV0103', 'v_dc_LV0104', 'i_sq_ref_LV0104', 'i_sd_ref_LV0104', 'i_sr_LV0104', 'i_si_LV0104', 'p_s_LV0104', 'q_s_LV0104', 'v_dc_LV0105', 'i_sq_ref_LV0105', 'i_sd_ref_LV0105', 'i_sr_LV0105', 'i_si_LV0105', 'p_s_LV0105', 'q_s_LV0105', 'v_dc_LV0201', 'i_sq_ref_LV0201', 'i_sd_ref_LV0201', 'i_sr_LV0201', 'i_si_LV0201', 'p_s_LV0201', 'q_s_LV0201', 'v_dc_LV0202', 'i_sq_ref_LV0202', 'i_sd_ref_LV0202', 'i_sr_LV0202', 'i_si_LV0202', 'p_s_LV0202', 'q_s_LV0202', 'v_dc_LV0203', 'i_sq_ref_LV0203', 'i_sd_ref_LV0203', 'i_sr_LV0203', 'i_si_LV0203', 'p_s_LV0203', 'q_s_LV0203', 'v_dc_LV0204', 'i_sq_ref_LV0204', 'i_sd_ref_LV0204', 'i_sr_LV0204', 'i_si_LV0204', 'p_s_LV0204', 'q_s_LV0204', 'v_dc_LV0205', 'i_sq_ref_LV0205', 'i_sd_ref_LV0205', 'i_sr_LV0205', 'i_si_LV0205', 'p_s_LV0205', 'q_s_LV0205', 'v_dc_LV0301', 'i_sq_ref_LV0301', 'i_sd_ref_LV0301', 'i_sr_LV0301', 'i_si_LV0301', 'p_s_LV0301', 'q_s_LV0301', 'v_dc_LV0302', 'i_sq_ref_LV0302', 'i_sd_ref_LV0302', 'i_sr_LV0302', 'i_si_LV0302', 'p_s_LV0302', 'q_s_LV0302', 'v_dc_LV0303', 'i_sq_ref_LV0303', 'i_sd_ref_LV0303', 'i_sr_LV0303', 'i_si_LV0303', 'p_s_LV0303', 'q_s_LV0303', 'v_dc_LV0304', 'i_sq_ref_LV0304', 'i_sd_ref_LV0304', 'i_sr_LV0304', 'i_si_LV0304', 'p_s_LV0304', 'q_s_LV0304', 'v_dc_LV0305', 'i_sq_ref_LV0305', 'i_sd_ref_LV0305', 'i_sr_LV0305', 'i_si_LV0305', 'p_s_LV0305', 'q_s_LV0305', 'v_dc_LV0401', 'i_sq_ref_LV0401', 'i_sd_ref_LV0401', 'i_sr_LV0401', 'i_si_LV0401', 'p_s_LV0401', 'q_s_LV0401', 'v_dc_LV0402', 'i_sq_ref_LV0402', 'i_sd_ref_LV0402', 'i_sr_LV0402', 'i_si_LV0402', 'p_s_LV0402', 'q_s_LV0402', 'v_dc_LV0403', 'i_sq_ref_LV0403', 'i_sd_ref_LV0403', 'i_sr_LV0403', 'i_si_LV0403', 'p_s_LV0403', 'q_s_LV0403', 'v_dc_LV0404', 'i_sq_ref_LV0404', 'i_sd_ref_LV0404', 'i_sr_LV0404', 'i_si_LV0404', 'p_s_LV0404', 'q_s_LV0404', 'v_dc_LV0405', 'i_sq_ref_LV0405', 'i_sd_ref_LV0405', 'i_sr_LV0405', 'i_si_LV0405', 'p_s_LV0405', 'q_s_LV0405', 'omega_coi', 'p_agc'] 
        self.xy_ini_list = self.x_list + self.y_ini_list 
        self.t = 0.0
        self.it = 0
        self.it_store = 0
        self.xy_prev = np.zeros((self.N_x+self.N_y,1))
        self.initialization_tol = 1e-6
        self.N_u = len(self.inputs_run_list) 
        self.sopt_root_method='hybr'
        self.sopt_root_jac=True
        self.u_ini_list = self.inputs_ini_list
        self.u_ini_values_list = self.inputs_ini_values_list
        self.u_run_list = self.inputs_run_list
        self.u_run_values_list = self.inputs_run_values_list
        self.N_u = len(self.u_run_list)
        self.u_ini = np.array(self.inputs_ini_values_list,dtype=np.float64)
        self.p = np.array(self.params_values_list,dtype=np.float64)
        self.xy_0 = np.zeros((self.N_x+self.N_y,),dtype=np.float64)
        self.xy = np.zeros((self.N_x+self.N_y,),dtype=np.float64)
        self.z = np.zeros((self.N_z,),dtype=np.float64)
        
        # numerical elements of jacobians computing:
        x = self.xy[:self.N_x]
        y = self.xy[self.N_x:]
        
        self.yini2urun = list(set(self.u_run_list).intersection(set(self.y_ini_list)))
        self.uini2yrun = list(set(self.y_run_list).intersection(set(self.u_ini_list)))
        self.Time = np.zeros(self.N_store)
        self.X = np.zeros((self.N_store,self.N_x))
        self.Y = np.zeros((self.N_store,self.N_y))
        self.Z = np.zeros((self.N_store,self.N_z))
        self.iters = np.zeros(self.N_store) 
        self.u_run = np.array(self.u_run_values_list,dtype=np.float64)
 
        ## jac_ini
        self.jac_ini = np.zeros((self.N_x+self.N_y,self.N_x+self.N_y))
        self.sp_jac_ini_ia, self.sp_jac_ini_ja, self.sp_jac_ini_nia, self.sp_jac_ini_nja = sp_jac_ini_vectors()
        data = np.array(self.sp_jac_ini_ia,dtype=np.float64)
        #self.sp_jac_ini = sspa.csr_matrix((data, self.sp_jac_ini_ia, self.sp_jac_ini_ja), shape=(self.sp_jac_ini_nia,self.sp_jac_ini_nja))
           
        if self.dae_file_mode == 'enviroment':
            fobj = BytesIO(pkgutil.get_data(__name__, f'./pv_4_5_sp_jac_ini_num.npz'))
            self.sp_jac_ini = sspa.load_npz(fobj)
        else:
            self.sp_jac_ini = sspa.load_npz(f'{self.matrices_folder}/pv_4_5_sp_jac_ini_num.npz')
            
            
        self.jac_ini = self.sp_jac_ini.toarray()

        #self.J_ini_d = np.array(self.sp_jac_ini_ia)*0.0
        #self.J_ini_i = np.array(self.sp_jac_ini_ia)
        #self.J_ini_p = np.array(self.sp_jac_ini_ja)
        de_jac_ini_eval(self.jac_ini,x,y,self.u_ini,self.p,self.Dt)
        sp_jac_ini_eval(self.sp_jac_ini.data,x,y,self.u_ini,self.p,self.Dt) 
        self.fill_factor_ini,self.drop_tol_ini,self.drop_rule_ini = 100,1e-10,'basic'       


        ## jac_run
        self.jac_run = np.zeros((self.N_x+self.N_y,self.N_x+self.N_y))
        self.sp_jac_run_ia, self.sp_jac_run_ja, self.sp_jac_run_nia, self.sp_jac_run_nja = sp_jac_run_vectors()
        data = np.array(self.sp_jac_run_ia,dtype=np.float64)

        if self.dae_file_mode == 'enviroment':
            fobj = BytesIO(pkgutil.get_data(__name__, './pv_4_5_sp_jac_run_num.npz'))
            self.sp_jac_run = sspa.load_npz(fobj)
        else:
            self.sp_jac_run = sspa.load_npz(f'{self.matrices_folder}/pv_4_5_sp_jac_run_num.npz')
        self.jac_run = self.sp_jac_run.toarray()            
           
        self.J_run_d = np.array(self.sp_jac_run_ia)*0.0
        self.J_run_i = np.array(self.sp_jac_run_ia)
        self.J_run_p = np.array(self.sp_jac_run_ja)
        de_jac_run_eval(self.jac_run,x,y,self.u_run,self.p,self.Dt)
        sp_jac_run_eval(self.J_run_d,x,y,self.u_run,self.p,self.Dt)
        
        ## jac_trap
        self.jac_trap = np.zeros((self.N_x+self.N_y,self.N_x+self.N_y))
        self.sp_jac_trap_ia, self.sp_jac_trap_ja, self.sp_jac_trap_nia, self.sp_jac_trap_nja = sp_jac_trap_vectors()
        data = np.array(self.sp_jac_trap_ia,dtype=np.float64)
        #self.sp_jac_trap = sspa.csr_matrix((data, self.sp_jac_trap_ia, self.sp_jac_trap_ja), shape=(self.sp_jac_trap_nia,self.sp_jac_trap_nja))
       
    

        if self.dae_file_mode == 'enviroment':
            fobj = BytesIO(pkgutil.get_data(__name__, './pv_4_5_sp_jac_trap_num.npz'))
            self.sp_jac_trap = sspa.load_npz(fobj)
        else:
            self.sp_jac_trap = sspa.load_npz(f'{self.matrices_folder}/pv_4_5_sp_jac_trap_num.npz')
            

        self.jac_trap = self.sp_jac_trap.toarray()
        
        #self.J_trap_d = np.array(self.sp_jac_trap_ia)*0.0
        #self.J_trap_i = np.array(self.sp_jac_trap_ia)
        #self.J_trap_p = np.array(self.sp_jac_trap_ja)
        de_jac_trap_eval(self.jac_trap,x,y,self.u_run,self.p,self.Dt)
        sp_jac_trap_eval(self.sp_jac_trap.data,x,y,self.u_run,self.p,self.Dt)
        self.fill_factor_trap,self.drop_tol_trap,self.drop_rule_trap = 100,1e-10,'basic' 
   

        

        
        self.max_it,self.itol,self.store = 50,1e-8,1 
        self.lmax_it,self.ltol,self.ldamp= 50,1e-8,1.0
        self.mode = 0 

        self.lmax_it_ini,self.ltol_ini,self.ldamp_ini=50,1e-8,1.0

        self.sp_Fu_run = sspa.load_npz(f'{self.matrices_folder}/pv_4_5_Fu_run_num.npz')
        self.sp_Gu_run = sspa.load_npz(f'{self.matrices_folder}/pv_4_5_Gu_run_num.npz')
        self.sp_Hx_run = sspa.load_npz(f'{self.matrices_folder}/pv_4_5_Hx_run_num.npz')
        self.sp_Hy_run = sspa.load_npz(f'{self.matrices_folder}/pv_4_5_Hy_run_num.npz')
        self.sp_Hu_run = sspa.load_npz(f'{self.matrices_folder}/pv_4_5_Hu_run_num.npz')        
        
        self.ss_solver = 2
        self.lsolver = 2
 
        



        
    def update(self):

        self.Time = np.zeros(self.N_store)
        self.X = np.zeros((self.N_store,self.N_x))
        self.Y = np.zeros((self.N_store,self.N_y))
        self.Z = np.zeros((self.N_store,self.N_z))
        self.iters = np.zeros(self.N_store)
        
    def ss_ini(self):

        xy_ini,it = sstate(self.xy_0,self.u_ini,self.p,self.jac_ini,self.N_x,self.N_y)
        self.xy_ini = xy_ini
        self.N_iters = it
        
        return xy_ini
    
    # def ini(self,up_dict,xy_0={}):

    #     for item in up_dict:
    #         self.set_value(item,up_dict[item])
            
    #     self.xy_ini = self.ss_ini()
    #     self.ini2run()
    #     jac_run_ss_eval_xy(self.jac_run,self.x,self.y_run,self.u_run,self.p)
    #     jac_run_ss_eval_up(self.jac_run,self.x,self.y_run,self.u_run,self.p)
        
    def jac_run_eval(self):
        de_jac_run_eval(self.jac_run,self.x,self.y_run,self.u_run,self.p,self.Dt)
      
    
    def run(self,t_end,up_dict):
        for item in up_dict:
            self.set_value(item,up_dict[item])
            
        t = self.t
        p = self.p
        it = self.it
        it_store = self.it_store
        xy = self.xy
        u = self.u_run
        z = self.z
        
        t,it,it_store,xy = daesolver(t,t_end,it,it_store,xy,u,p,z,
                                  self.jac_trap,
                                  self.Time,
                                  self.X,
                                  self.Y,
                                  self.Z,
                                  self.iters,
                                  self.Dt,
                                  self.N_x,
                                  self.N_y,
                                  self.N_z,
                                  self.decimation,
                                  max_it=self.max_it,itol=self.itol,store=self.store)
        
        self.t = t
        self.it = it
        self.it_store = it_store
        self.xy = xy
        self.z = z
 
    def runsp(self,t_end,up_dict):
        for item in up_dict:
            self.set_value(item,up_dict[item])
            
        t = self.t
        p = self.p
        it = self.it
        it_store = self.it_store
        xy = self.xy
        u = self.u_run
        
        t,it,it_store,xy = daesolver_sp(t,t_end,it,it_store,xy,u,p,
                                  self.sp_jac_trap,
                                  self.Time,
                                  self.X,
                                  self.Y,
                                  self.Z,
                                  self.iters,
                                  self.Dt,
                                  self.N_x,
                                  self.N_y,
                                  self.N_z,
                                  self.decimation,
                                  max_it=50,itol=1e-8,store=1)
        
        self.t = t
        self.it = it
        self.it_store = it_store
        self.xy = xy
        
    def post(self):
        
        self.Time = self.Time[:self.it_store]
        self.X = self.X[:self.it_store]
        self.Y = self.Y[:self.it_store]
        self.Z = self.Z[:self.it_store]
        
    def ini2run(self):
        
        ## y_ini to y_run
        self.y_ini = self.xy_ini[self.N_x:]
        self.y_run = np.copy(self.y_ini)
        #self.u_run = np.copy(self.u_ini)
        
        ## y_ini to u_run
        for item in self.yini2urun:
            self.u_run[self.u_run_list.index(item)] = self.y_ini[self.y_ini_list.index(item)]
                
        ## u_ini to y_run
        for item in self.uini2yrun:
            self.y_run[self.y_run_list.index(item)] = self.u_ini[self.u_ini_list.index(item)]
            
        
        self.x = self.xy_ini[:self.N_x]
        self.xy[:self.N_x] = self.x
        self.xy[self.N_x:] = self.y_run
        c_h_eval(self.z,self.x,self.y_run,self.u_ini,self.p,self.Dt)
        

        
    def get_value(self,name):
        
        if name in self.inputs_run_list:
            value = self.u_run[self.inputs_run_list.index(name)]
            return value
            
        if name in self.x_list:
            idx = self.x_list.index(name)
            value = self.xy[idx]
            return value
            
        if name in self.y_run_list:
            idy = self.y_run_list.index(name)
            value = self.xy[self.N_x+idy]
            return value
        
        if name in self.params_list:
            idp = self.params_list.index(name)
            value = self.p[idp]
            return value
            
        if name in self.outputs_list:
            idz = self.outputs_list.index(name)
            value = self.z[idz]
            return value

    def get_values(self,name):
        if name in self.x_list:
            values = self.X[:,self.x_list.index(name)]
        if name in self.y_run_list:
            values = self.Y[:,self.y_run_list.index(name)]
        if name in self.outputs_list:
            values = self.Z[:,self.outputs_list.index(name)]
                        
        return values

    def get_mvalue(self,names):
        '''

        Parameters
        ----------
        names : list
            list of variables names to return each value.

        Returns
        -------
        mvalue : TYPE
            list of value of each variable.

        '''
        mvalue = []
        for name in names:
            mvalue += [self.get_value(name)]
                        
        return mvalue
    
    def set_value(self,name_,value):
        if name_ in self.inputs_ini_list or name_ in self.inputs_run_list:
            if name_ in self.inputs_ini_list:
                self.u_ini[self.inputs_ini_list.index(name_)] = value
            if name_ in self.inputs_run_list:
                self.u_run[self.inputs_run_list.index(name_)] = value
            return
        elif name_ in self.params_list:
            self.p[self.params_list.index(name_)] = value
            return
        else:
            print(f'Input or parameter {name_} not found.')
 
    def report_x(self,value_format='5.2f'):
        for item in self.x_list:
            print(f'{item:5s} = {self.get_value(item):{value_format}}')

    def report_y(self,value_format='5.2f'):
        for item in self.y_run_list:
            print(f'{item:5s} = {self.get_value(item):{value_format}}')
            
    def report_u(self,value_format='5.2f'):
        for item in self.inputs_run_list:
            print(f'{item:5s} ={self.get_value(item):{value_format}}')

    def report_z(self,value_format='5.2f'):
        for item in self.outputs_list:
            print(f'{item:5s} = {self.get_value(item):{value_format}}')

    def report_params(self,value_format='5.2f'):
        for item in self.params_list:
            print(f'{item:5s} ={self.get_value(item):{value_format}}')
            
    def ini(self,up_dict,xy_0={}):
        '''
        Find the steady state of the initialization problem:
            
               0 = f(x,y,u,p) 
               0 = g(x,y,u,p) 

        Parameters
        ----------
        up_dict : dict
            dictionary with all the parameters p and inputs u new values.
        xy_0: if scalar, all the x and y values initial guess are set to the scalar.
              if dict, the initial guesses are applied for the x and y that are in the dictionary
              if string, the initial guess considers a json file with the x and y names and their initial values

        Returns
        -------
        mvalue : TYPE
            list of value of each variable.

        '''
        
        self.it = 0
        self.it_store = 0
        self.t = 0.0
    
        for item in up_dict:
            self.set_value(item,up_dict[item])
            
        if type(xy_0) == dict:
            xy_0_dict = xy_0
            self.dict2xy0(xy_0_dict)
            
        if type(xy_0) == str:
            if xy_0 == 'eval':
                N_x = self.N_x
                self.xy_0_new = np.copy(self.xy_0)*0
                xy0_eval(self.xy_0_new[:N_x],self.xy_0_new[N_x:],self.u_ini,self.p)
                self.xy_0_evaluated = np.copy(self.xy_0_new)
                self.xy_0 = np.copy(self.xy_0_new)
            else:
                self.load_xy_0(file_name = xy_0)
                
        if type(xy_0) == float or type(xy_0) == int:
            self.xy_0 = np.ones(self.N_x+self.N_y,dtype=np.float64)*xy_0

        xy_ini,it = sstate(self.xy_0,self.u_ini,self.p,
                           self.jac_ini,
                           self.N_x,self.N_y,
                           max_it=self.max_it,tol=self.itol)
        
        if it < self.max_it-1:
            
            self.xy_ini = xy_ini
            self.N_iters = it

            self.ini2run()
            
            self.ini_convergence = True
            
        if it >= self.max_it-1:
            print(f'Maximum number of iterations (max_it = {self.max_it}) reached without convergence.')
            self.ini_convergence = False
            
        return self.ini_convergence
            
        


    
    def dict2xy0(self,xy_0_dict):
    
        for item in xy_0_dict:
            if item in self.x_list:
                self.xy_0[self.x_list.index(item)] = xy_0_dict[item]
            if item in self.y_ini_list:
                self.xy_0[self.y_ini_list.index(item) + self.N_x] = xy_0_dict[item]
        
    
    def save_xy_0(self,file_name = 'xy_0.json'):
        xy_0_dict = {}
        for item in self.x_list:
            xy_0_dict.update({item:self.get_value(item)})
        for item in self.y_ini_list:
            xy_0_dict.update({item:self.get_value(item)})
    
        xy_0_str = json.dumps(xy_0_dict, indent=4)
        with open(file_name,'w') as fobj:
            fobj.write(xy_0_str)
    
    def load_xy_0(self,file_name = 'xy_0.json'):
        with open(file_name) as fobj:
            xy_0_str = fobj.read()
        xy_0_dict = json.loads(xy_0_str)
    
        for item in xy_0_dict:
            if item in self.x_list:
                self.xy_0[self.x_list.index(item)] = xy_0_dict[item]
            if item in self.y_ini_list:
                self.xy_0[self.y_ini_list.index(item)+self.N_x] = xy_0_dict[item]            

    def load_params(self,data_input):
    
        if type(data_input) == str:
            json_file = data_input
            self.json_file = json_file
            self.json_data = open(json_file).read().replace("'",'"')
            data = json.loads(self.json_data)
        elif type(data_input) == dict:
            data = data_input
    
        self.data = data
        for item in self.data:
            self.set_value(item, self.data[item])

    def save_params(self,file_name = 'parameters.json'):
        params_dict = {}
        for item in self.params_list:
            params_dict.update({item:self.get_value(item)})

        params_dict_str = json.dumps(params_dict, indent=4)
        with open(file_name,'w') as fobj:
            fobj.write(params_dict_str)

    def save_inputs_ini(self,file_name = 'inputs_ini.json'):
        inputs_ini_dict = {}
        for item in self.inputs_ini_list:
            inputs_ini_dict.update({item:self.get_value(item)})

        inputs_ini_dict_str = json.dumps(inputs_ini_dict, indent=4)
        with open(file_name,'w') as fobj:
            fobj.write(inputs_ini_dict_str)

    def eval_preconditioner_ini(self):
    
        sp_jac_ini_eval(self.sp_jac_ini.data,self.x,self.y_run,self.u_run,self.p,self.Dt)
    
        csc_sp_jac_ini = sspa.csc_matrix(self.sp_jac_ini)
        P_slu = spilu(csc_sp_jac_ini,
                  fill_factor=self.fill_factor_ini,
                  drop_tol=self.drop_tol_ini,
                  drop_rule = self.drop_rule_ini)
    
        self.P_slu = P_slu
        P_d,P_i,P_p,perm_r,perm_c = slu2pydae(P_slu)   
        self.P_d = P_d
        self.P_i = P_i
        self.P_p = P_p
    
        self.perm_r = perm_r
        self.perm_c = perm_c
            
    
    def eval_preconditioner_trap(self):
    
        sp_jac_trap_eval(self.sp_jac_trap.data,self.x,self.y_run,self.u_run,self.p,self.Dt)
    
        #self.sp_jac_trap.data = self.J_trap_d 
        
        csc_sp_jac_trap = sspa.csc_matrix(self.sp_jac_trap)


        P_slu_trap = spilu(csc_sp_jac_trap,
                          fill_factor=self.fill_factor_trap,
                          drop_tol=self.drop_tol_trap,
                          drop_rule = self.drop_rule_trap)
    
        self.P_slu_trap = P_slu_trap
        P_d,P_i,P_p,perm_r,perm_c = slu2pydae(P_slu_trap)   
        self.P_trap_d = P_d
        self.P_trap_i = P_i
        self.P_trap_p = P_p
    
        self.perm_trap_r = perm_r
        self.perm_trap_c = perm_c
        
    def sprun(self,t_end,up_dict):
        
        for item in up_dict:
            self.set_value(item,up_dict[item])
    
        t = self.t
        p = self.p
        it = self.it
        it_store = self.it_store
        xy = self.xy
        u = self.u_run
        z = self.z
        self.iparams_run = np.zeros(10,dtype=np.float64)
    
        t,it,it_store,xy = spdaesolver(t,t_end,it,it_store,xy,u,p,z,
                                  self.sp_jac_trap.data,self.sp_jac_trap.indices,self.sp_jac_trap.indptr,
                                  self.P_trap_d,self.P_trap_i,self.P_trap_p,self.perm_trap_r,self.perm_trap_c,
                                  self.Time,
                                  self.X,
                                  self.Y,
                                  self.Z,
                                  self.iters,
                                  self.Dt,
                                  self.N_x,
                                  self.N_y,
                                  self.N_z,
                                  self.decimation,
                                  self.iparams_run,
                                  max_it=self.max_it,itol=self.max_it,store=self.store,
                                  lmax_it=self.lmax_it,ltol=self.ltol,ldamp=self.ldamp,mode=self.mode,
                                  lsolver = self.lsolver)
    
        self.t = t
        self.it = it
        self.it_store = it_store
        self.xy = xy
        self.z = z

            
    def spini(self,up_dict,xy_0={}):
    
        self.it = 0
        self.it_store = 0
        self.t = 0.0
    
        for item in up_dict:
            self.set_value(item,up_dict[item])
    
        if type(xy_0) == dict:
            xy_0_dict = xy_0
            self.dict2xy0(xy_0_dict)
    
        if type(xy_0) == str:
            if xy_0 == 'eval':
                N_x = self.N_x
                self.xy_0_new = np.copy(self.xy_0)*0
                xy0_eval(self.xy_0_new[:N_x],self.xy_0_new[N_x:],self.u_ini,self.p)
                self.xy_0_evaluated = np.copy(self.xy_0_new)
                self.xy_0 = np.copy(self.xy_0_new)
            else:
                self.load_xy_0(file_name = xy_0)

        self.xy_ini = self.spss_ini()


        if self.N_iters < self.max_it:
            
            self.ini2run()           
            self.ini_convergence = True
            
        if self.N_iters >= self.max_it:
            print(f'Maximum number of iterations (max_it = {self.max_it}) reached without convergence.')
            self.ini_convergence = False
            
        #jac_run_eval_xy(self.jac_run,self.x,self.y_run,self.u_run,self.p)
        #jac_run_eval_up(self.jac_run,self.x,self.y_run,self.u_run,self.p)
        
        return self.ini_convergence

        
    def spss_ini(self):
        J_d,J_i,J_p = csr2pydae(self.sp_jac_ini)
        
        xy_ini,it,iparams = spsstate(self.xy,self.u_ini,self.p,
                 self.sp_jac_ini.data,self.sp_jac_ini.indices,self.sp_jac_ini.indptr,
                 self.P_d,self.P_i,self.P_p,self.perm_r,self.perm_c,
                 self.N_x,self.N_y,
                 max_it=self.max_it,tol=self.itol,
                 lmax_it=self.lmax_it_ini,
                 ltol=self.ltol_ini,
                 ldamp=self.ldamp,solver=self.ss_solver)

 
        self.xy_ini = xy_ini
        self.N_iters = it
        self.iparams = iparams
    
        return xy_ini

    #def import_cffi(self):
        

    def eval_jac_u2z(self):

        '''

        0 =   J_run * xy + FG_u * u
        z = Hxy_run * xy + H_u * u

        xy = -1/J_run * FG_u * u
        z = -Hxy_run/J_run * FG_u * u + H_u * u
        z = (-Hxy_run/J_run * FG_u + H_u ) * u 
        '''
        
        sp_Fu_run_eval(self.sp_Fu_run.data,self.x,self.y_run,self.u_run,self.p,self.Dt)
        sp_Gu_run_eval(self.sp_Gu_run.data,self.x,self.y_run,self.u_run,self.p,self.Dt)
        sp_H_jacs_run_eval(self.sp_Hx_run.data,
                        self.sp_Hy_run.data,
                        self.sp_Hu_run.data,
                        self.x,self.y_run,self.u_run,self.p,self.Dt)
        sp_jac_run = self.sp_jac_run
        sp_jac_run_eval(sp_jac_run.data,
                        self.x,self.y_run,
                        self.u_run,self.p,
                        self.Dt)



        Hxy_run = sspa.bmat([[self.sp_Hx_run,self.sp_Hy_run]])
        FGu_run = sspa.bmat([[self.sp_Fu_run],[self.sp_Gu_run]])
        

        #((sspa.linalg.spsolve(s.sp_jac_ini,-Hxy_run)) @ FGu_run + sp_Hu_run )@s.u_ini

        self.jac_u2z = Hxy_run @ sspa.linalg.spsolve(self.sp_jac_run,-FGu_run) + self.sp_Hu_run  
        
        
    def step(self,t_end,up_dict):
        for item in up_dict:
            self.set_value(item,up_dict[item])

        t = self.t
        p = self.p
        it = self.it
        it_store = self.it_store
        xy = self.xy
        u = self.u_run
        z = self.z

        t,it,xy = daestep(t,t_end,it,
                          xy,u,p,z,
                          self.jac_trap,
                          self.iters,
                          self.Dt,
                          self.N_x,
                          self.N_y,
                          self.N_z,
                          max_it=self.max_it,itol=self.itol,store=self.store)

        self.t = t
        self.it = it
        self.it_store = it_store
        self.xy = xy
        self.z = z
           
            
    def save_run(self,file_name):
        np.savez(file_name,Time=self.Time,
             X=self.X,Y=self.Y,Z=self.Z,
             x_list = self.x_list,
             y_ini_list = self.y_ini_list,
             y_run_list = self.y_run_list,
             u_ini_list=self.u_ini_list,
             u_run_list=self.u_run_list,  
             z_list=self.outputs_list, 
            )
        
    def load_run(self,file_name):
        data = np.load(f'{file_name}.npz')
        self.Time = data['Time']
        self.X = data['X']
        self.Y = data['Y']
        self.Z = data['Z']
        self.x_list = list(data['x_list'] )
        self.y_run_list = list(data['y_run_list'] )
        self.outputs_list = list(data['z_list'] )
        
    def full_jacs_eval(self):
        N_x = self.N_x
        N_y = self.N_y
        N_xy = N_x + N_y
    
        sp_jac_run = self.sp_jac_run
        sp_Fu = self.sp_Fu_run
        sp_Gu = self.sp_Gu_run
        sp_Hx = self.sp_Hx_run
        sp_Hy = self.sp_Hy_run
        sp_Hu = self.sp_Hu_run
        
        x = self.xy[0:N_x]
        y = self.xy[N_x:]
        u = self.u_run
        p = self.p
        Dt = self.Dt
    
        sp_jac_run_eval(sp_jac_run.data,x,y,u,p,Dt)
        
        self.Fx = sp_jac_run[0:N_x,0:N_x]
        self.Fy = sp_jac_run[ 0:N_x,N_x:]
        self.Gx = sp_jac_run[ N_x:,0:N_x]
        self.Gy = sp_jac_run[ N_x:, N_x:]
        
        sp_Fu_run_eval(sp_Fu.data,x,y,u,p,Dt)
        sp_Gu_run_eval(sp_Gu.data,x,y,u,p,Dt)
        sp_H_jacs_run_eval(sp_Hx.data,sp_Hy.data,sp_Hu.data,x,y,u,p,Dt)
        
        self.Fu = sp_Fu
        self.Gu = sp_Gu
        self.Hx = sp_Hx
        self.Hy = sp_Hy
        self.Hu = sp_Hu


@numba.njit() 
def daestep(t,t_end,it,xy,u,p,z,jac_trap,iters,Dt,N_x,N_y,N_z,max_it=50,itol=1e-8,store=1): 


    fg = np.zeros((N_x+N_y,1),dtype=np.float64)
    fg_i = np.zeros((N_x+N_y),dtype=np.float64)
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]
    #h = np.zeros((N_z),dtype=np.float64)
    
    f_ptr=ffi.from_buffer(np.ascontiguousarray(f))
    g_ptr=ffi.from_buffer(np.ascontiguousarray(g))
    z_ptr=ffi.from_buffer(np.ascontiguousarray(z))
    x_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    jac_trap_ptr=ffi.from_buffer(np.ascontiguousarray(jac_trap))
    
    #de_jac_trap_num_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)    
    de_jac_trap_up_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    de_jac_trap_xy_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    
    if it == 0:
        f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        h_eval(z_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        it_store = 0  

    while t<t_end: 
        it += 1
        t += Dt

        f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)

        x_0 = np.copy(x) 
        y_0 = np.copy(y) 
        f_0 = np.copy(f) 
        g_0 = np.copy(g) 
            
        for iti in range(max_it):
            f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
            g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
            de_jac_trap_xy_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 

            f_n_i = x - x_0 - 0.5*Dt*(f+f_0) 

            fg_i[:N_x] = f_n_i
            fg_i[N_x:] = g
            
            Dxy_i = np.linalg.solve(-jac_trap,fg_i) 

            x += Dxy_i[:N_x]
            y += Dxy_i[N_x:] 
            
            #print(Dxy_i)

            # iteration stop
            max_relative = 0.0
            for it_var in range(N_x+N_y):
                abs_value = np.abs(xy[it_var])
                if abs_value < 0.001:
                    abs_value = 0.001
                relative_error = np.abs(Dxy_i[it_var])/abs_value

                if relative_error > max_relative: max_relative = relative_error

            if max_relative<itol:
                break
                
        h_eval(z_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        xy[:N_x] = x
        xy[N_x:] = y
        
    return t,it,xy


def daesolver_sp(t,t_end,it,it_store,xy,u,p,sp_jac_trap,T,X,Y,Z,iters,Dt,N_x,N_y,N_z,decimation,max_it=50,itol=1e-8,store=1): 

    fg = np.zeros((N_x+N_y,1),dtype=np.float64)
    fg_i = np.zeros((N_x+N_y),dtype=np.float64)
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]
    h = np.zeros((N_z),dtype=np.float64)
    sp_jac_trap_eval_up(sp_jac_trap.data,x,y,u,p,Dt,xyup=1)
    
    if it == 0:
        f_run_eval(f,x,y,u,p)
        h_eval(h,x,y,u,p)
        it_store = 0  
        T[0] = t 
        X[0,:] = x  
        Y[0,:] = y  
        Z[0,:] = h  

    while t<t_end: 
        it += 1
        t += Dt

        f_run_eval(f,x,y,u,p)
        g_run_eval(g,x,y,u,p)

        x_0 = np.copy(x) 
        y_0 = np.copy(y) 
        f_0 = np.copy(f) 
        g_0 = np.copy(g) 
            
        for iti in range(max_it):
            f_run_eval(f,x,y,u,p)
            g_run_eval(g,x,y,u,p)
            sp_jac_trap_eval(sp_jac_trap.data,x,y,u,p,Dt,xyup=1)            

            f_n_i = x - x_0 - 0.5*Dt*(f+f_0) 

            fg_i[:N_x] = f_n_i
            fg_i[N_x:] = g
            
            Dxy_i = spsolve(sp_jac_trap,-fg_i) 

            x = x + Dxy_i[:N_x]
            y = y + Dxy_i[N_x:]              

            # iteration stop
            max_relative = 0.0
            for it_var in range(N_x+N_y):
                abs_value = np.abs(xy[it_var])
                if abs_value < 0.001:
                    abs_value = 0.001
                relative_error = np.abs(Dxy_i[it_var])/abs_value

                if relative_error > max_relative: max_relative = relative_error

            if max_relative<itol:
                break
                
        h_eval(h,x,y,u,p)
        xy[:N_x] = x
        xy[N_x:] = y
        
        # store in channels 
        if store == 1:
            if it >= it_store*decimation: 
                T[it_store+1] = t 
                X[it_store+1,:] = x 
                Y[it_store+1,:] = y
                Z[it_store+1,:] = h
                iters[it_store+1] = iti
                it_store += 1 

    return t,it,it_store,xy




@numba.njit()
def sprichardson(A_d,A_i,A_p,b,P_d,P_i,P_p,perm_r,perm_c,x,iparams,damp=1.0,max_it=100,tol=1e-3):
    N_A = A_p.shape[0]-1
    f = np.zeros(N_A)
    for it in range(max_it):
        spMvmul(N_A,A_d,A_i,A_p,x,f) 
        f -= b                          # A@x-b
        x = x - damp*splu_solve(P_d,P_i,P_p,perm_r,perm_c,f)   
        if np.linalg.norm(f,2) < tol: break
    iparams[0] = it
    return x
    
@numba.njit()
def spconjgradm(A_d,A_i,A_p,b,P_d,P_i,P_p,perm_r,perm_c,x,iparams,max_it=100,tol=1e-3, damp=None):
    """
    A function to solve [A]{x} = {b} linear equation system with the 
    preconditioned conjugate gradient method.
    More at: http://en.wikipedia.org/wiki/Conjugate_gradient_method
    ========== Parameters ==========
    A_d,A_i,A_p : sparse matrix 
        components in CRS form A_d = A_crs.data, A_i = A_crs.indices, A_p = A_crs.indptr.
    b : vector
        The right hand side (RHS) vector of the system.
    x : vector
        The starting guess for the solution.
    P_d,P_i,P_p,perm_r,perm_c: preconditioner LU matrix
        components in scipy.spilu form P_d,P_i,P_p,perm_r,perm_c = slu2pydae(M)
        with M = scipy.sparse.linalg.spilu(A_csc) 

    """  
    N   = len(b)
    Ax  = np.zeros(N)
    Ap  = np.zeros(N)
    App = np.zeros(N)
    pAp = np.zeros(N)
    z   = np.zeros(N)
    
    spMvmul(N,A_d,A_i,A_p,x,Ax)
    r = -(Ax - b)
    z = splu_solve(P_d,P_i,P_p,perm_r,perm_c,r) #z = M.solve(r)
    p = z
    zsold = 0.0
    for it in range(N):  # zsold = np.dot(np.transpose(z), z)
        zsold += z[it]*z[it]
    for i in range(max_it):
        spMvmul(N,A_d,A_i,A_p,p,App)  # #App = np.dot(A, p)
        Ap = splu_solve(P_d,P_i,P_p,perm_r,perm_c,App) #Ap = M.solve(App)
        pAp = 0.0
        for it in range(N):
            pAp += p[it]*Ap[it]

        alpha = zsold / pAp
        x = x + alpha*p
        z = z - alpha*Ap
        zz = 0.0
        for it in range(N):  # z.T@z
            zz += z[it]*z[it]
        zsnew = zz
        if np.sqrt(zsnew) < tol:
            break
            
        p = z + (zsnew/zsold)*p
        zsold = zsnew
    iparams[0] = i

    return x


@numba.njit()
def spsstate(xy,u,p,
             J_d,J_i,J_p,
             P_d,P_i,P_p,perm_r,perm_c,
             N_x,N_y,
             max_it=50,tol=1e-8,
             lmax_it=20,ltol=1e-8,ldamp=1.0, solver=2):
    
   
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]
    iparams = np.array([0],dtype=np.int64)    
    
    f_c_ptr=ffi.from_buffer(np.ascontiguousarray(f))
    g_c_ptr=ffi.from_buffer(np.ascontiguousarray(g))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))
    J_d_ptr=ffi.from_buffer(np.ascontiguousarray(J_d))

    #sp_jac_ini_num_eval(J_d_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
    sp_jac_ini_up_eval(J_d_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
    
    #sp_jac_ini_eval_up(J_d,x,y,u,p,0.0)

    Dxy = np.zeros(N_x + N_y)
    for it in range(max_it):
        
        x = xy[:N_x]
        y = xy[N_x:]   
       
        sp_jac_ini_xy_eval(J_d_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)

        
        f_ini_eval(f_c_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
        g_ini_eval(g_c_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
        
        #f_ini_eval(f,x,y,u,p)
        #g_ini_eval(g,x,y,u,p)
        
        fg[:N_x] = f
        fg[N_x:] = g
        
        if solver==1:
               
            Dxy = sprichardson(J_d,J_i,J_p,-fg,P_d,P_i,P_p,perm_r,perm_c,Dxy,iparams,damp=ldamp,max_it=lmax_it,tol=ltol)
   
        if solver==2:
            
            Dxy = spconjgradm(J_d,J_i,J_p,-fg,P_d,P_i,P_p,perm_r,perm_c,Dxy,iparams,damp=ldamp,max_it=lmax_it,tol=ltol)
            
        xy += Dxy
        #if np.max(np.abs(fg))<tol: break
        if np.linalg.norm(fg,np.inf)<tol: break

    return xy,it,iparams


    
@numba.njit() 
def daesolver(t,t_end,it,it_store,xy,u,p,z,jac_trap,T,X,Y,Z,iters,Dt,N_x,N_y,N_z,decimation,max_it=50,itol=1e-8,store=1): 


    fg = np.zeros((N_x+N_y,1),dtype=np.float64)
    fg_i = np.zeros((N_x+N_y),dtype=np.float64)
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]
    #h = np.zeros((N_z),dtype=np.float64)
    
    f_ptr=ffi.from_buffer(np.ascontiguousarray(f))
    g_ptr=ffi.from_buffer(np.ascontiguousarray(g))
    z_ptr=ffi.from_buffer(np.ascontiguousarray(z))
    x_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    jac_trap_ptr=ffi.from_buffer(np.ascontiguousarray(jac_trap))
    
    #de_jac_trap_num_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)    
    de_jac_trap_up_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    de_jac_trap_xy_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    
    if it == 0:
        f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        h_eval(z_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        it_store = 0  
        T[0] = t 
        X[0,:] = x  
        Y[0,:] = y  
        Z[0,:] = z  

    while t<t_end: 
        it += 1
        t += Dt

        f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)

        x_0 = np.copy(x) 
        y_0 = np.copy(y) 
        f_0 = np.copy(f) 
        g_0 = np.copy(g) 
            
        for iti in range(max_it):
            f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
            g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
            de_jac_trap_xy_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 

            f_n_i = x - x_0 - 0.5*Dt*(f+f_0) 

            fg_i[:N_x] = f_n_i
            fg_i[N_x:] = g
            
            Dxy_i = np.linalg.solve(-jac_trap,fg_i) 

            x += Dxy_i[:N_x]
            y += Dxy_i[N_x:] 
            
            #print(Dxy_i)

            # iteration stop
            max_relative = 0.0
            for it_var in range(N_x+N_y):
                abs_value = np.abs(xy[it_var])
                if abs_value < 0.001:
                    abs_value = 0.001
                relative_error = np.abs(Dxy_i[it_var])/abs_value

                if relative_error > max_relative: max_relative = relative_error

            if max_relative<itol:
                break
                
        h_eval(z_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        xy[:N_x] = x
        xy[N_x:] = y
        
        # store in channels 
        if store == 1:
            if it >= it_store*decimation: 
                T[it_store+1] = t 
                X[it_store+1,:] = x 
                Y[it_store+1,:] = y
                Z[it_store+1,:] = z
                iters[it_store+1] = iti
                it_store += 1 

    return t,it,it_store,xy
    
@numba.njit() 
def spdaesolver(t,t_end,it,it_store,xy,u,p,z,
                J_d,J_i,J_p,
                P_d,P_i,P_p,perm_r,perm_c,
                T,X,Y,Z,iters,Dt,N_x,N_y,N_z,decimation,
                iparams,
                max_it=50,itol=1e-8,store=1,
                lmax_it=20,ltol=1e-4,ldamp=1.0,mode=0,lsolver=2):

    fg_i = np.zeros((N_x+N_y),dtype=np.float64)
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]
    z = np.zeros((N_z),dtype=np.float64)
    Dxy_i_0 = np.zeros(N_x+N_y,dtype=np.float64) 
    f_ptr=ffi.from_buffer(np.ascontiguousarray(f))
    g_ptr=ffi.from_buffer(np.ascontiguousarray(g))
    z_ptr=ffi.from_buffer(np.ascontiguousarray(z))
    x_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    J_d_ptr=ffi.from_buffer(np.ascontiguousarray(J_d))
    
    #sp_jac_trap_num_eval(J_d_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)    
    sp_jac_trap_up_eval( J_d_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    sp_jac_trap_xy_eval( J_d_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    
    if it == 0:
        f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        h_eval(z_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        it_store = 0  
        T[0] = t 
        X[0,:] = x  
        Y[0,:] = y  
        Z[0,:] = z 

    while t<t_end: 
        it += 1
        t += Dt

        f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)

        x_0 = np.copy(x) 
        y_0 = np.copy(y) 
        f_0 = np.copy(f) 
        g_0 = np.copy(g) 
            
        for iti in range(max_it):
            f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
            g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
            sp_jac_trap_xy_eval(J_d_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 

            f_n_i = x - x_0 - 0.5*Dt*(f+f_0) 

            fg_i[:N_x] = f_n_i
            fg_i[N_x:] = g
            
            #Dxy_i = np.linalg.solve(-jac_trap,fg_i) 
            if lsolver == 1:
                Dxy_i = sprichardson(J_d,J_i,J_p,-fg_i,P_d,P_i,P_p,perm_r,perm_c,
                                     Dxy_i_0,iparams,damp=ldamp,max_it=lmax_it,tol=ltol)
            if lsolver == 2:
                Dxy_i = spconjgradm(J_d,J_i,J_p,-fg_i,P_d,P_i,P_p,perm_r,perm_c,
                                     Dxy_i_0,iparams,damp=ldamp,max_it=lmax_it,tol=ltol)                

            x += Dxy_i[:N_x]
            y += Dxy_i[N_x:] 
            
            #print(Dxy_i)

            # iteration stop
            max_relative = 0.0
            for it_var in range(N_x+N_y):
                abs_value = np.abs(xy[it_var])
                if abs_value < 0.001:
                    abs_value = 0.001
                relative_error = np.abs(Dxy_i[it_var])/abs_value

                if relative_error > max_relative: max_relative = relative_error

            if max_relative<itol:
                break
                
        h_eval(z_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        xy[:N_x] = x
        xy[N_x:] = y
        
        # store in channels 
        if store == 1:
            if it >= it_store*decimation: 
                T[it_store+1] = t 
                X[it_store+1,:] = x 
                Y[it_store+1,:] = y
                Z[it_store+1,:] = z
                iters[it_store+1] = iti
                it_store += 1 

    return t,it,it_store,xy


@cuda.jit()
def ode_solve(x,u,p,f_run,u_idxs,z_i,z,sim):

    N_i,N_j,N_x,N_z,Dt = sim

    # index of thread on GPU:
    i = cuda.grid(1)

    if i < x.size:
        for j in range(N_j):
            f_run_eval(f_run[i,:],x[i,:],u[i,u_idxs[j],:],p[i,:])
            for k in range(N_x):
              x[i,k] +=  Dt*f_run[i,k]

            # outputs in time range
            #z[i,j] = u[i,idxs[j],0]
            z[i,j] = x[i,1]
        h_eval(z_i[i,:],x[i,:],u[i,u_idxs[j],:],p[i,:])
        
def csr2pydae(A_csr):
    '''
    From scipy CSR to the three vectors:
    
    - data
    - indices
    - indptr
    
    '''
    
    A_d = A_csr.data
    A_i = A_csr.indices
    A_p = A_csr.indptr
    
    return A_d,A_i,A_p
    
def slu2pydae(P_slu):
    '''
    From SupderLU matrix to the three vectors:
    
    - data
    - indices
    - indptr
    
    and the premutation vectors:
    
    - perm_r
    - perm_c
    
    '''
    N = P_slu.shape[0]
    #P_slu_full = P_slu.L.A - sspa.eye(N,format='csr') + P_slu.U.A
    P_slu_full = P_slu.L - sspa.eye(N,format='csc') + P_slu.U
    perm_r = P_slu.perm_r
    perm_c = P_slu.perm_c
    P_csr = sspa.csr_matrix(P_slu_full)
    
    P_d = P_csr.data
    P_i = P_csr.indices
    P_p = P_csr.indptr
    
    return P_d,P_i,P_p,perm_r,perm_c

@numba.njit(cache=True)
def spMvmul(N,A_data,A_indices,A_indptr,x,y):
    '''
    y = A @ x
    
    with A in sparse CRS form
    '''
    #y = np.zeros(x.shape[0])
    for i in range(N):
        y[i] = 0.0
        for j in range(A_indptr[i],A_indptr[i + 1]):
            y[i] = y[i] + A_data[j]*x[A_indices[j]]
            
            
@numba.njit(cache=True)
def splu_solve(LU_d,LU_i,LU_p,perm_r,perm_c,b):
    N = len(b)
    y = np.zeros(N)
    x = np.zeros(N)
    z = np.zeros(N)
    bp = np.zeros(N)
    
    for i in range(N): 
        bp[perm_r[i]] = b[i]
        
    for i in range(N): 
        y[i] = bp[i]
        for j in range(LU_p[i],LU_p[i+1]):
            if LU_i[j]>i-1: break
            y[i] -= LU_d[j] * y[LU_i[j]]

    for i in range(N-1,-1,-1): #(int i = N - 1; i >= 0; i--) 
        z[i] = y[i]
        den = 0.0
        for j in range(LU_p[i],LU_p[i+1]): #(int k = i + 1; k < N; k++)
            if LU_i[j] > i:
                z[i] -= LU_d[j] * z[LU_i[j]]
            if LU_i[j] == i: den = LU_d[j]
        z[i] = z[i]/den
 
    for i in range(N):
        x[i] = z[perm_c[i]]
        
    return x



@numba.njit("float64[:,:](float64[:,:],float64[:],float64[:],float64[:],float64[:],float64)")
def de_jac_ini_eval(de_jac_ini,x,y,u,p,Dt):   
    '''
    Computes the dense full initialization jacobian:
    
    jac_ini = [[Fx_ini, Fy_ini],
               [Gx_ini, Gy_ini]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    de_jac_ini : (N, N) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (ini problem).
    u : (N_u,) array_like
        Vector with inputs (ini problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with N = N_x+N_y
 
    Returns
    -------
    
    de_jac_ini : (N, N) array_like
                  Updated matrix.    
    
    '''
    
    de_jac_ini_ptr=ffi.from_buffer(np.ascontiguousarray(de_jac_ini))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    de_jac_ini_num_eval(de_jac_ini_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    de_jac_ini_up_eval( de_jac_ini_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    de_jac_ini_xy_eval( de_jac_ini_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    return de_jac_ini

@numba.njit("float64[:,:](float64[:,:],float64[:],float64[:],float64[:],float64[:],float64)")
def de_jac_run_eval(de_jac_run,x,y,u,p,Dt):   
    '''
    Computes the dense full initialization jacobian:
    
    jac_run = [[Fx_run, Fy_run],
               [Gx_run, Gy_run]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    de_jac_run : (N, N) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (ini problem).
    u : (N_u,) array_like
        Vector with inputs (ini problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with N = N_x+N_y
 
    Returns
    -------
    
    de_jac_ini : (N, N) array_like
                  Updated matrix.    
    
    '''
    
    de_jac_run_ptr=ffi.from_buffer(np.ascontiguousarray(de_jac_run))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    de_jac_run_num_eval(de_jac_run_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    de_jac_run_up_eval( de_jac_run_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    de_jac_run_xy_eval( de_jac_run_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    return de_jac_run

@numba.njit("float64[:,:](float64[:,:],float64[:],float64[:],float64[:],float64[:],float64)")
def de_jac_trap_eval(de_jac_trap,x,y,u,p,Dt):   
    '''
    Computes the dense full trapezoidal jacobian:
    
    jac_trap = [[eye - 0.5*Dt*Fx_run, -0.5*Dt*Fy_run],
                [             Gx_run,         Gy_run]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    de_jac_trap : (N, N) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (run problem).
    u : (N_u,) array_like
        Vector with inputs (run problem). 
    p : (N_p,) array_like
        Vector with parameters. 
 
    Returns
    -------
    
    de_jac_trap : (N, N) array_like
                  Updated matrix.    
    
    '''
        
    de_jac_trap_ptr = ffi.from_buffer(np.ascontiguousarray(de_jac_trap))
    x_c_ptr = ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr = ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr = ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr = ffi.from_buffer(np.ascontiguousarray(p))

    de_jac_trap_num_eval(de_jac_trap_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    de_jac_trap_up_eval( de_jac_trap_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    de_jac_trap_xy_eval( de_jac_trap_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    return de_jac_trap


@numba.njit("float64[:](float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
def sp_jac_run_eval(sp_jac_run,x,y,u,p,Dt):   
    '''
    Computes the sparse full trapezoidal jacobian:
    
    jac_trap = [[eye - 0.5*Dt*Fx_run, -0.5*Dt*Fy_run],
                [             Gx_run,         Gy_run]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    sp_jac_trap : (Nnz,) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (run problem).
    u : (N_u,) array_like
        Vector with inputs (run problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with Nnz the number of non-zeros elements in the jacobian.
 
    Returns
    -------
    
    sp_jac_trap : (Nnz,) array_like
                  Updated matrix.    
    
    '''        
    sp_jac_run_ptr=ffi.from_buffer(np.ascontiguousarray(sp_jac_run))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    sp_jac_run_num_eval( sp_jac_run_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_jac_run_up_eval( sp_jac_run_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_jac_run_xy_eval( sp_jac_run_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    return sp_jac_run

@numba.njit("float64[:](float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
def sp_jac_trap_eval(sp_jac_trap,x,y,u,p,Dt):   
    '''
    Computes the sparse full trapezoidal jacobian:
    
    jac_trap = [[eye - 0.5*Dt*Fx_run, -0.5*Dt*Fy_run],
                [             Gx_run,         Gy_run]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    sp_jac_trap : (Nnz,) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (run problem).
    u : (N_u,) array_like
        Vector with inputs (run problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with Nnz the number of non-zeros elements in the jacobian.
 
    Returns
    -------
    
    sp_jac_trap : (Nnz,) array_like
                  Updated matrix.    
    
    '''        
    sp_jac_trap_ptr=ffi.from_buffer(np.ascontiguousarray(sp_jac_trap))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    sp_jac_trap_num_eval(sp_jac_trap_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_jac_trap_up_eval( sp_jac_trap_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_jac_trap_xy_eval( sp_jac_trap_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    return sp_jac_trap

@numba.njit("float64[:](float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
def sp_jac_ini_eval(sp_jac_ini,x,y,u,p,Dt):   
    '''
    Computes the SPARSE full initialization jacobian:
    
    jac_ini = [[Fx_ini, Fy_ini],
               [Gx_ini, Gy_ini]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    de_jac_ini : (N, N) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (ini problem).
    u : (N_u,) array_like
        Vector with inputs (ini problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with N = N_x+N_y
 
    Returns
    -------
    
    de_jac_ini : (N, N) array_like
                  Updated matrix.    
    
    '''
    
    sp_jac_ini_ptr=ffi.from_buffer(np.ascontiguousarray(sp_jac_ini))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    sp_jac_ini_num_eval(sp_jac_ini_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_jac_ini_up_eval( sp_jac_ini_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_jac_ini_xy_eval( sp_jac_ini_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    return sp_jac_ini


@numba.njit()
def sstate(xy,u,p,jac_ini_ss,N_x,N_y,max_it=50,tol=1e-8):
    
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]

    f_c_ptr=ffi.from_buffer(np.ascontiguousarray(f))
    g_c_ptr=ffi.from_buffer(np.ascontiguousarray(g))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))
    jac_ini_ss_ptr=ffi.from_buffer(np.ascontiguousarray(jac_ini_ss))

    #de_jac_ini_num_eval(jac_ini_ss_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
    de_jac_ini_up_eval(jac_ini_ss_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)

    for it in range(max_it):
        de_jac_ini_xy_eval(jac_ini_ss_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
        f_ini_eval(f_c_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
        g_ini_eval(g_c_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
        fg[:N_x] = f
        fg[N_x:] = g
        xy += np.linalg.solve(jac_ini_ss,-fg)
        if np.max(np.abs(fg))<tol: break

    return xy,it


@numba.njit("float64[:](float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
def c_h_eval(z,x,y,u,p,Dt):   
    '''
    Computes the SPARSE full initialization jacobian:
    
    jac_ini = [[Fx_ini, Fy_ini],
               [Gx_ini, Gy_ini]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    de_jac_ini : (N, N) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (ini problem).
    u : (N_u,) array_like
        Vector with inputs (ini problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with N = N_x+N_y
 
    Returns
    -------
    
    de_jac_ini : (N, N) array_like
                  Updated matrix.    
    
    '''
    
    z_c_ptr=ffi.from_buffer(np.ascontiguousarray(z))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    h_eval(z_c_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    return z

@numba.njit("(float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
def sp_Fu_run_eval(jac,x,y,u,p,Dt):   
    '''
    Computes the dense full initialization jacobian:
    
    jac_ini = [[Fx_ini, Fy_ini],
               [Gx_ini, Gy_ini]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    de_jac_ini : (N, N) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (ini problem).
    u : (N_u,) array_like
        Vector with inputs (ini problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with N = N_x+N_y
 
    Returns
    -------
    
    de_jac_ini : (N, N) array_like
                  Updated matrix.    
    
    '''
    
    jac_ptr=ffi.from_buffer(np.ascontiguousarray(jac))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    sp_Fu_run_up_eval( jac_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_Fu_run_xy_eval( jac_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    #return jac

@numba.njit("(float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
def sp_Gu_run_eval(jac,x,y,u,p,Dt):   
    '''
    Computes the dense full initialization jacobian:
    
    jac_ini = [[Fx_ini, Fy_ini],
               [Gx_ini, Gy_ini]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    de_jac_ini : (N, N) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (ini problem).
    u : (N_u,) array_like
        Vector with inputs (ini problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with N = N_x+N_y
 
    Returns
    -------
    
    de_jac_ini : (N, N) array_like
                  Updated matrix.    
    
    '''
    
    jac_ptr=ffi.from_buffer(np.ascontiguousarray(jac))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    sp_Gu_run_up_eval( jac_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_Gu_run_xy_eval( jac_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    #return jac

@numba.njit("(float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
def sp_H_jacs_run_eval(H_x,H_y,H_u,x,y,u,p,Dt):   
    '''
    Computes the dense full initialization jacobian:
    
    jac_ini = [[Fx_ini, Fy_ini],
               [Gx_ini, Gy_ini]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    de_jac_ini : (N, N) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (ini problem).
    u : (N_u,) array_like
        Vector with inputs (ini problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with N = N_x+N_y
 
    Returns
    -------
    
    de_jac_ini : (N, N) array_like
                  Updated matrix.    
    
    '''
    
    H_x_ptr=ffi.from_buffer(np.ascontiguousarray(H_x))
    H_y_ptr=ffi.from_buffer(np.ascontiguousarray(H_y))
    H_u_ptr=ffi.from_buffer(np.ascontiguousarray(H_u))

    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    sp_Hx_run_up_eval( H_x_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_Hx_run_xy_eval( H_x_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_Hy_run_up_eval( H_y_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_Hy_run_xy_eval( H_y_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_Hu_run_up_eval( H_u_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_Hu_run_xy_eval( H_u_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)





def sp_jac_ini_vectors():

    sp_jac_ini_ia = [0, 175, 0, 2, 177, 322, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 13, 14, 15, 15, 16, 17, 17, 18, 19, 19, 20, 21, 21, 22, 23, 23, 24, 25, 25, 26, 27, 27, 28, 29, 29, 30, 31, 31, 32, 33, 33, 34, 35, 35, 36, 37, 37, 38, 39, 39, 40, 41, 41, 42, 43, 43, 44, 45, 45, 46, 47, 47, 48, 49, 49, 50, 51, 51, 52, 53, 53, 54, 55, 55, 56, 57, 57, 58, 59, 59, 60, 61, 61, 62, 63, 63, 64, 65, 65, 66, 67, 67, 68, 69, 69, 70, 71, 71, 72, 73, 73, 74, 75, 75, 76, 77, 77, 78, 79, 79, 80, 81, 81, 82, 83, 83, 84, 85, 322, 86, 87, 88, 89, 92, 93, 96, 97, 116, 117, 136, 137, 156, 157, 86, 87, 88, 89, 92, 93, 96, 97, 116, 117, 136, 137, 156, 157, 86, 87, 88, 89, 90, 91, 86, 87, 88, 89, 90, 91, 88, 89, 90, 91, 180, 88, 89, 90, 91, 181, 0, 1, 86, 87, 92, 93, 86, 87, 92, 93, 94, 95, 96, 97, 187, 94, 95, 96, 97, 188, 86, 87, 94, 95, 96, 97, 100, 101, 86, 87, 94, 95, 96, 97, 100, 101, 98, 99, 100, 101, 194, 98, 99, 100, 101, 195, 96, 97, 98, 99, 100, 101, 104, 105, 96, 97, 98, 99, 100, 101, 104, 105, 102, 103, 104, 105, 201, 102, 103, 104, 105, 202, 100, 101, 102, 103, 104, 105, 108, 109, 100, 101, 102, 103, 104, 105, 108, 109, 106, 107, 108, 109, 208, 106, 107, 108, 109, 209, 104, 105, 106, 107, 108, 109, 112, 113, 104, 105, 106, 107, 108, 109, 112, 113, 110, 111, 112, 113, 215, 110, 111, 112, 113, 216, 108, 109, 110, 111, 112, 113, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 222, 114, 115, 116, 117, 223, 86, 87, 114, 115, 116, 117, 120, 121, 86, 87, 114, 115, 116, 117, 120, 121, 118, 119, 120, 121, 229, 118, 119, 120, 121, 230, 116, 117, 118, 119, 120, 121, 124, 125, 116, 117, 118, 119, 120, 121, 124, 125, 122, 123, 124, 125, 236, 122, 123, 124, 125, 237, 120, 121, 122, 123, 124, 125, 128, 129, 120, 121, 122, 123, 124, 125, 128, 129, 126, 127, 128, 129, 243, 126, 127, 128, 129, 244, 124, 125, 126, 127, 128, 129, 132, 133, 124, 125, 126, 127, 128, 129, 132, 133, 130, 131, 132, 133, 250, 130, 131, 132, 133, 251, 128, 129, 130, 131, 132, 133, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 257, 134, 135, 136, 137, 258, 86, 87, 134, 135, 136, 137, 140, 141, 86, 87, 134, 135, 136, 137, 140, 141, 138, 139, 140, 141, 264, 138, 139, 140, 141, 265, 136, 137, 138, 139, 140, 141, 144, 145, 136, 137, 138, 139, 140, 141, 144, 145, 142, 143, 144, 145, 271, 142, 143, 144, 145, 272, 140, 141, 142, 143, 144, 145, 148, 149, 140, 141, 142, 143, 144, 145, 148, 149, 146, 147, 148, 149, 278, 146, 147, 148, 149, 279, 144, 145, 146, 147, 148, 149, 152, 153, 144, 145, 146, 147, 148, 149, 152, 153, 150, 151, 152, 153, 285, 150, 151, 152, 153, 286, 148, 149, 150, 151, 152, 153, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 292, 154, 155, 156, 157, 293, 86, 87, 154, 155, 156, 157, 160, 161, 86, 87, 154, 155, 156, 157, 160, 161, 158, 159, 160, 161, 299, 158, 159, 160, 161, 300, 156, 157, 158, 159, 160, 161, 164, 165, 156, 157, 158, 159, 160, 161, 164, 165, 162, 163, 164, 165, 306, 162, 163, 164, 165, 307, 160, 161, 162, 163, 164, 165, 168, 169, 160, 161, 162, 163, 164, 165, 168, 169, 166, 167, 168, 169, 313, 166, 167, 168, 169, 314, 164, 165, 166, 167, 168, 169, 172, 173, 164, 165, 166, 167, 168, 169, 172, 173, 170, 171, 172, 173, 320, 170, 171, 172, 173, 321, 168, 169, 170, 171, 172, 173, 168, 169, 170, 171, 172, 173, 0, 1, 92, 174, 174, 175, 176, 0, 175, 176, 3, 177, 2, 90, 91, 178, 179, 2, 4, 90, 91, 178, 179, 2, 90, 91, 178, 179, 180, 2, 90, 91, 178, 179, 181, 182, 187, 8, 94, 184, 6, 94, 183, 94, 95, 183, 184, 185, 186, 94, 95, 183, 184, 185, 186, 94, 95, 185, 186, 187, 94, 95, 185, 186, 188, 189, 194, 12, 98, 191, 10, 98, 190, 98, 99, 190, 191, 192, 193, 98, 99, 190, 191, 192, 193, 98, 99, 192, 193, 194, 98, 99, 192, 193, 195, 196, 201, 16, 102, 198, 14, 102, 197, 102, 103, 197, 198, 199, 200, 102, 103, 197, 198, 199, 200, 102, 103, 199, 200, 201, 102, 103, 199, 200, 202, 203, 208, 20, 106, 205, 18, 106, 204, 106, 107, 204, 205, 206, 207, 106, 107, 204, 205, 206, 207, 106, 107, 206, 207, 208, 106, 107, 206, 207, 209, 210, 215, 24, 110, 212, 22, 110, 211, 110, 111, 211, 212, 213, 214, 110, 111, 211, 212, 213, 214, 110, 111, 213, 214, 215, 110, 111, 213, 214, 216, 217, 222, 28, 114, 219, 26, 114, 218, 114, 115, 218, 219, 220, 221, 114, 115, 218, 219, 220, 221, 114, 115, 220, 221, 222, 114, 115, 220, 221, 223, 224, 229, 32, 118, 226, 30, 118, 225, 118, 119, 225, 226, 227, 228, 118, 119, 225, 226, 227, 228, 118, 119, 227, 228, 229, 118, 119, 227, 228, 230, 231, 236, 36, 122, 233, 34, 122, 232, 122, 123, 232, 233, 234, 235, 122, 123, 232, 233, 234, 235, 122, 123, 234, 235, 236, 122, 123, 234, 235, 237, 238, 243, 40, 126, 240, 38, 126, 239, 126, 127, 239, 240, 241, 242, 126, 127, 239, 240, 241, 242, 126, 127, 241, 242, 243, 126, 127, 241, 242, 244, 245, 250, 44, 130, 247, 42, 130, 246, 130, 131, 246, 247, 248, 249, 130, 131, 246, 247, 248, 249, 130, 131, 248, 249, 250, 130, 131, 248, 249, 251, 252, 257, 48, 134, 254, 46, 134, 253, 134, 135, 253, 254, 255, 256, 134, 135, 253, 254, 255, 256, 134, 135, 255, 256, 257, 134, 135, 255, 256, 258, 259, 264, 52, 138, 261, 50, 138, 260, 138, 139, 260, 261, 262, 263, 138, 139, 260, 261, 262, 263, 138, 139, 262, 263, 264, 138, 139, 262, 263, 265, 266, 271, 56, 142, 268, 54, 142, 267, 142, 143, 267, 268, 269, 270, 142, 143, 267, 268, 269, 270, 142, 143, 269, 270, 271, 142, 143, 269, 270, 272, 273, 278, 60, 146, 275, 58, 146, 274, 146, 147, 274, 275, 276, 277, 146, 147, 274, 275, 276, 277, 146, 147, 276, 277, 278, 146, 147, 276, 277, 279, 280, 285, 64, 150, 282, 62, 150, 281, 150, 151, 281, 282, 283, 284, 150, 151, 281, 282, 283, 284, 150, 151, 283, 284, 285, 150, 151, 283, 284, 286, 287, 292, 68, 154, 289, 66, 154, 288, 154, 155, 288, 289, 290, 291, 154, 155, 288, 289, 290, 291, 154, 155, 290, 291, 292, 154, 155, 290, 291, 293, 294, 299, 72, 158, 296, 70, 158, 295, 158, 159, 295, 296, 297, 298, 158, 159, 295, 296, 297, 298, 158, 159, 297, 298, 299, 158, 159, 297, 298, 300, 301, 306, 76, 162, 303, 74, 162, 302, 162, 163, 302, 303, 304, 305, 162, 163, 302, 303, 304, 305, 162, 163, 304, 305, 306, 162, 163, 304, 305, 307, 308, 313, 80, 166, 310, 78, 166, 309, 166, 167, 309, 310, 311, 312, 166, 167, 309, 310, 311, 312, 166, 167, 311, 312, 313, 166, 167, 311, 312, 314, 315, 320, 84, 170, 317, 82, 170, 316, 170, 171, 316, 317, 318, 319, 170, 171, 316, 317, 318, 319, 170, 171, 318, 319, 320, 170, 171, 318, 319, 321, 177, 322, 85, 322, 323]
    sp_jac_ini_ja = [0, 2, 3, 6, 7, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29, 30, 32, 33, 35, 36, 38, 39, 41, 42, 44, 45, 47, 48, 50, 51, 53, 54, 56, 57, 59, 60, 62, 63, 65, 66, 68, 69, 71, 72, 74, 75, 77, 78, 80, 81, 83, 84, 86, 87, 89, 90, 92, 93, 95, 96, 98, 99, 101, 102, 104, 105, 107, 108, 110, 111, 113, 114, 116, 117, 119, 120, 122, 123, 125, 126, 128, 130, 144, 158, 164, 170, 175, 180, 186, 190, 195, 200, 208, 216, 221, 226, 234, 242, 247, 252, 260, 268, 273, 278, 286, 294, 299, 304, 310, 316, 321, 326, 334, 342, 347, 352, 360, 368, 373, 378, 386, 394, 399, 404, 412, 420, 425, 430, 436, 442, 447, 452, 460, 468, 473, 478, 486, 494, 499, 504, 512, 520, 525, 530, 538, 546, 551, 556, 562, 568, 573, 578, 586, 594, 599, 604, 612, 620, 625, 630, 638, 646, 651, 656, 664, 672, 677, 682, 688, 694, 698, 701, 704, 706, 711, 717, 723, 729, 731, 734, 737, 743, 749, 754, 759, 761, 764, 767, 773, 779, 784, 789, 791, 794, 797, 803, 809, 814, 819, 821, 824, 827, 833, 839, 844, 849, 851, 854, 857, 863, 869, 874, 879, 881, 884, 887, 893, 899, 904, 909, 911, 914, 917, 923, 929, 934, 939, 941, 944, 947, 953, 959, 964, 969, 971, 974, 977, 983, 989, 994, 999, 1001, 1004, 1007, 1013, 1019, 1024, 1029, 1031, 1034, 1037, 1043, 1049, 1054, 1059, 1061, 1064, 1067, 1073, 1079, 1084, 1089, 1091, 1094, 1097, 1103, 1109, 1114, 1119, 1121, 1124, 1127, 1133, 1139, 1144, 1149, 1151, 1154, 1157, 1163, 1169, 1174, 1179, 1181, 1184, 1187, 1193, 1199, 1204, 1209, 1211, 1214, 1217, 1223, 1229, 1234, 1239, 1241, 1244, 1247, 1253, 1259, 1264, 1269, 1271, 1274, 1277, 1283, 1289, 1294, 1299, 1301, 1304, 1307, 1313, 1319, 1324, 1329, 1331, 1334]
    sp_jac_ini_nia = 324
    sp_jac_ini_nja = 324
    return sp_jac_ini_ia, sp_jac_ini_ja, sp_jac_ini_nia, sp_jac_ini_nja 

def sp_jac_run_vectors():

    sp_jac_run_ia = [0, 175, 0, 2, 177, 322, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 13, 14, 15, 15, 16, 17, 17, 18, 19, 19, 20, 21, 21, 22, 23, 23, 24, 25, 25, 26, 27, 27, 28, 29, 29, 30, 31, 31, 32, 33, 33, 34, 35, 35, 36, 37, 37, 38, 39, 39, 40, 41, 41, 42, 43, 43, 44, 45, 45, 46, 47, 47, 48, 49, 49, 50, 51, 51, 52, 53, 53, 54, 55, 55, 56, 57, 57, 58, 59, 59, 60, 61, 61, 62, 63, 63, 64, 65, 65, 66, 67, 67, 68, 69, 69, 70, 71, 71, 72, 73, 73, 74, 75, 75, 76, 77, 77, 78, 79, 79, 80, 81, 81, 82, 83, 83, 84, 85, 322, 86, 87, 88, 89, 92, 93, 96, 97, 116, 117, 136, 137, 156, 157, 86, 87, 88, 89, 92, 93, 96, 97, 116, 117, 136, 137, 156, 157, 86, 87, 88, 89, 90, 91, 86, 87, 88, 89, 90, 91, 88, 89, 90, 91, 180, 88, 89, 90, 91, 181, 0, 1, 86, 87, 92, 93, 86, 87, 92, 93, 94, 95, 96, 97, 187, 94, 95, 96, 97, 188, 86, 87, 94, 95, 96, 97, 100, 101, 86, 87, 94, 95, 96, 97, 100, 101, 98, 99, 100, 101, 194, 98, 99, 100, 101, 195, 96, 97, 98, 99, 100, 101, 104, 105, 96, 97, 98, 99, 100, 101, 104, 105, 102, 103, 104, 105, 201, 102, 103, 104, 105, 202, 100, 101, 102, 103, 104, 105, 108, 109, 100, 101, 102, 103, 104, 105, 108, 109, 106, 107, 108, 109, 208, 106, 107, 108, 109, 209, 104, 105, 106, 107, 108, 109, 112, 113, 104, 105, 106, 107, 108, 109, 112, 113, 110, 111, 112, 113, 215, 110, 111, 112, 113, 216, 108, 109, 110, 111, 112, 113, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 222, 114, 115, 116, 117, 223, 86, 87, 114, 115, 116, 117, 120, 121, 86, 87, 114, 115, 116, 117, 120, 121, 118, 119, 120, 121, 229, 118, 119, 120, 121, 230, 116, 117, 118, 119, 120, 121, 124, 125, 116, 117, 118, 119, 120, 121, 124, 125, 122, 123, 124, 125, 236, 122, 123, 124, 125, 237, 120, 121, 122, 123, 124, 125, 128, 129, 120, 121, 122, 123, 124, 125, 128, 129, 126, 127, 128, 129, 243, 126, 127, 128, 129, 244, 124, 125, 126, 127, 128, 129, 132, 133, 124, 125, 126, 127, 128, 129, 132, 133, 130, 131, 132, 133, 250, 130, 131, 132, 133, 251, 128, 129, 130, 131, 132, 133, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 257, 134, 135, 136, 137, 258, 86, 87, 134, 135, 136, 137, 140, 141, 86, 87, 134, 135, 136, 137, 140, 141, 138, 139, 140, 141, 264, 138, 139, 140, 141, 265, 136, 137, 138, 139, 140, 141, 144, 145, 136, 137, 138, 139, 140, 141, 144, 145, 142, 143, 144, 145, 271, 142, 143, 144, 145, 272, 140, 141, 142, 143, 144, 145, 148, 149, 140, 141, 142, 143, 144, 145, 148, 149, 146, 147, 148, 149, 278, 146, 147, 148, 149, 279, 144, 145, 146, 147, 148, 149, 152, 153, 144, 145, 146, 147, 148, 149, 152, 153, 150, 151, 152, 153, 285, 150, 151, 152, 153, 286, 148, 149, 150, 151, 152, 153, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 292, 154, 155, 156, 157, 293, 86, 87, 154, 155, 156, 157, 160, 161, 86, 87, 154, 155, 156, 157, 160, 161, 158, 159, 160, 161, 299, 158, 159, 160, 161, 300, 156, 157, 158, 159, 160, 161, 164, 165, 156, 157, 158, 159, 160, 161, 164, 165, 162, 163, 164, 165, 306, 162, 163, 164, 165, 307, 160, 161, 162, 163, 164, 165, 168, 169, 160, 161, 162, 163, 164, 165, 168, 169, 166, 167, 168, 169, 313, 166, 167, 168, 169, 314, 164, 165, 166, 167, 168, 169, 172, 173, 164, 165, 166, 167, 168, 169, 172, 173, 170, 171, 172, 173, 320, 170, 171, 172, 173, 321, 168, 169, 170, 171, 172, 173, 168, 169, 170, 171, 172, 173, 0, 1, 92, 174, 174, 175, 176, 0, 175, 176, 3, 177, 2, 90, 91, 178, 179, 2, 4, 90, 91, 178, 179, 2, 90, 91, 178, 179, 180, 2, 90, 91, 178, 179, 181, 182, 187, 8, 94, 184, 6, 94, 183, 94, 95, 183, 184, 185, 186, 94, 95, 183, 184, 185, 186, 94, 95, 185, 186, 187, 94, 95, 185, 186, 188, 189, 194, 12, 98, 191, 10, 98, 190, 98, 99, 190, 191, 192, 193, 98, 99, 190, 191, 192, 193, 98, 99, 192, 193, 194, 98, 99, 192, 193, 195, 196, 201, 16, 102, 198, 14, 102, 197, 102, 103, 197, 198, 199, 200, 102, 103, 197, 198, 199, 200, 102, 103, 199, 200, 201, 102, 103, 199, 200, 202, 203, 208, 20, 106, 205, 18, 106, 204, 106, 107, 204, 205, 206, 207, 106, 107, 204, 205, 206, 207, 106, 107, 206, 207, 208, 106, 107, 206, 207, 209, 210, 215, 24, 110, 212, 22, 110, 211, 110, 111, 211, 212, 213, 214, 110, 111, 211, 212, 213, 214, 110, 111, 213, 214, 215, 110, 111, 213, 214, 216, 217, 222, 28, 114, 219, 26, 114, 218, 114, 115, 218, 219, 220, 221, 114, 115, 218, 219, 220, 221, 114, 115, 220, 221, 222, 114, 115, 220, 221, 223, 224, 229, 32, 118, 226, 30, 118, 225, 118, 119, 225, 226, 227, 228, 118, 119, 225, 226, 227, 228, 118, 119, 227, 228, 229, 118, 119, 227, 228, 230, 231, 236, 36, 122, 233, 34, 122, 232, 122, 123, 232, 233, 234, 235, 122, 123, 232, 233, 234, 235, 122, 123, 234, 235, 236, 122, 123, 234, 235, 237, 238, 243, 40, 126, 240, 38, 126, 239, 126, 127, 239, 240, 241, 242, 126, 127, 239, 240, 241, 242, 126, 127, 241, 242, 243, 126, 127, 241, 242, 244, 245, 250, 44, 130, 247, 42, 130, 246, 130, 131, 246, 247, 248, 249, 130, 131, 246, 247, 248, 249, 130, 131, 248, 249, 250, 130, 131, 248, 249, 251, 252, 257, 48, 134, 254, 46, 134, 253, 134, 135, 253, 254, 255, 256, 134, 135, 253, 254, 255, 256, 134, 135, 255, 256, 257, 134, 135, 255, 256, 258, 259, 264, 52, 138, 261, 50, 138, 260, 138, 139, 260, 261, 262, 263, 138, 139, 260, 261, 262, 263, 138, 139, 262, 263, 264, 138, 139, 262, 263, 265, 266, 271, 56, 142, 268, 54, 142, 267, 142, 143, 267, 268, 269, 270, 142, 143, 267, 268, 269, 270, 142, 143, 269, 270, 271, 142, 143, 269, 270, 272, 273, 278, 60, 146, 275, 58, 146, 274, 146, 147, 274, 275, 276, 277, 146, 147, 274, 275, 276, 277, 146, 147, 276, 277, 278, 146, 147, 276, 277, 279, 280, 285, 64, 150, 282, 62, 150, 281, 150, 151, 281, 282, 283, 284, 150, 151, 281, 282, 283, 284, 150, 151, 283, 284, 285, 150, 151, 283, 284, 286, 287, 292, 68, 154, 289, 66, 154, 288, 154, 155, 288, 289, 290, 291, 154, 155, 288, 289, 290, 291, 154, 155, 290, 291, 292, 154, 155, 290, 291, 293, 294, 299, 72, 158, 296, 70, 158, 295, 158, 159, 295, 296, 297, 298, 158, 159, 295, 296, 297, 298, 158, 159, 297, 298, 299, 158, 159, 297, 298, 300, 301, 306, 76, 162, 303, 74, 162, 302, 162, 163, 302, 303, 304, 305, 162, 163, 302, 303, 304, 305, 162, 163, 304, 305, 306, 162, 163, 304, 305, 307, 308, 313, 80, 166, 310, 78, 166, 309, 166, 167, 309, 310, 311, 312, 166, 167, 309, 310, 311, 312, 166, 167, 311, 312, 313, 166, 167, 311, 312, 314, 315, 320, 84, 170, 317, 82, 170, 316, 170, 171, 316, 317, 318, 319, 170, 171, 316, 317, 318, 319, 170, 171, 318, 319, 320, 170, 171, 318, 319, 321, 177, 322, 85, 322, 323]
    sp_jac_run_ja = [0, 2, 3, 6, 7, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29, 30, 32, 33, 35, 36, 38, 39, 41, 42, 44, 45, 47, 48, 50, 51, 53, 54, 56, 57, 59, 60, 62, 63, 65, 66, 68, 69, 71, 72, 74, 75, 77, 78, 80, 81, 83, 84, 86, 87, 89, 90, 92, 93, 95, 96, 98, 99, 101, 102, 104, 105, 107, 108, 110, 111, 113, 114, 116, 117, 119, 120, 122, 123, 125, 126, 128, 130, 144, 158, 164, 170, 175, 180, 186, 190, 195, 200, 208, 216, 221, 226, 234, 242, 247, 252, 260, 268, 273, 278, 286, 294, 299, 304, 310, 316, 321, 326, 334, 342, 347, 352, 360, 368, 373, 378, 386, 394, 399, 404, 412, 420, 425, 430, 436, 442, 447, 452, 460, 468, 473, 478, 486, 494, 499, 504, 512, 520, 525, 530, 538, 546, 551, 556, 562, 568, 573, 578, 586, 594, 599, 604, 612, 620, 625, 630, 638, 646, 651, 656, 664, 672, 677, 682, 688, 694, 698, 701, 704, 706, 711, 717, 723, 729, 731, 734, 737, 743, 749, 754, 759, 761, 764, 767, 773, 779, 784, 789, 791, 794, 797, 803, 809, 814, 819, 821, 824, 827, 833, 839, 844, 849, 851, 854, 857, 863, 869, 874, 879, 881, 884, 887, 893, 899, 904, 909, 911, 914, 917, 923, 929, 934, 939, 941, 944, 947, 953, 959, 964, 969, 971, 974, 977, 983, 989, 994, 999, 1001, 1004, 1007, 1013, 1019, 1024, 1029, 1031, 1034, 1037, 1043, 1049, 1054, 1059, 1061, 1064, 1067, 1073, 1079, 1084, 1089, 1091, 1094, 1097, 1103, 1109, 1114, 1119, 1121, 1124, 1127, 1133, 1139, 1144, 1149, 1151, 1154, 1157, 1163, 1169, 1174, 1179, 1181, 1184, 1187, 1193, 1199, 1204, 1209, 1211, 1214, 1217, 1223, 1229, 1234, 1239, 1241, 1244, 1247, 1253, 1259, 1264, 1269, 1271, 1274, 1277, 1283, 1289, 1294, 1299, 1301, 1304, 1307, 1313, 1319, 1324, 1329, 1331, 1334]
    sp_jac_run_nia = 324
    sp_jac_run_nja = 324
    return sp_jac_run_ia, sp_jac_run_ja, sp_jac_run_nia, sp_jac_run_nja 

def sp_jac_trap_vectors():

    sp_jac_trap_ia = [0, 175, 0, 1, 2, 177, 322, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 13, 14, 15, 15, 16, 17, 17, 18, 19, 19, 20, 21, 21, 22, 23, 23, 24, 25, 25, 26, 27, 27, 28, 29, 29, 30, 31, 31, 32, 33, 33, 34, 35, 35, 36, 37, 37, 38, 39, 39, 40, 41, 41, 42, 43, 43, 44, 45, 45, 46, 47, 47, 48, 49, 49, 50, 51, 51, 52, 53, 53, 54, 55, 55, 56, 57, 57, 58, 59, 59, 60, 61, 61, 62, 63, 63, 64, 65, 65, 66, 67, 67, 68, 69, 69, 70, 71, 71, 72, 73, 73, 74, 75, 75, 76, 77, 77, 78, 79, 79, 80, 81, 81, 82, 83, 83, 84, 85, 322, 86, 87, 88, 89, 92, 93, 96, 97, 116, 117, 136, 137, 156, 157, 86, 87, 88, 89, 92, 93, 96, 97, 116, 117, 136, 137, 156, 157, 86, 87, 88, 89, 90, 91, 86, 87, 88, 89, 90, 91, 88, 89, 90, 91, 180, 88, 89, 90, 91, 181, 0, 1, 86, 87, 92, 93, 86, 87, 92, 93, 94, 95, 96, 97, 187, 94, 95, 96, 97, 188, 86, 87, 94, 95, 96, 97, 100, 101, 86, 87, 94, 95, 96, 97, 100, 101, 98, 99, 100, 101, 194, 98, 99, 100, 101, 195, 96, 97, 98, 99, 100, 101, 104, 105, 96, 97, 98, 99, 100, 101, 104, 105, 102, 103, 104, 105, 201, 102, 103, 104, 105, 202, 100, 101, 102, 103, 104, 105, 108, 109, 100, 101, 102, 103, 104, 105, 108, 109, 106, 107, 108, 109, 208, 106, 107, 108, 109, 209, 104, 105, 106, 107, 108, 109, 112, 113, 104, 105, 106, 107, 108, 109, 112, 113, 110, 111, 112, 113, 215, 110, 111, 112, 113, 216, 108, 109, 110, 111, 112, 113, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 222, 114, 115, 116, 117, 223, 86, 87, 114, 115, 116, 117, 120, 121, 86, 87, 114, 115, 116, 117, 120, 121, 118, 119, 120, 121, 229, 118, 119, 120, 121, 230, 116, 117, 118, 119, 120, 121, 124, 125, 116, 117, 118, 119, 120, 121, 124, 125, 122, 123, 124, 125, 236, 122, 123, 124, 125, 237, 120, 121, 122, 123, 124, 125, 128, 129, 120, 121, 122, 123, 124, 125, 128, 129, 126, 127, 128, 129, 243, 126, 127, 128, 129, 244, 124, 125, 126, 127, 128, 129, 132, 133, 124, 125, 126, 127, 128, 129, 132, 133, 130, 131, 132, 133, 250, 130, 131, 132, 133, 251, 128, 129, 130, 131, 132, 133, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 257, 134, 135, 136, 137, 258, 86, 87, 134, 135, 136, 137, 140, 141, 86, 87, 134, 135, 136, 137, 140, 141, 138, 139, 140, 141, 264, 138, 139, 140, 141, 265, 136, 137, 138, 139, 140, 141, 144, 145, 136, 137, 138, 139, 140, 141, 144, 145, 142, 143, 144, 145, 271, 142, 143, 144, 145, 272, 140, 141, 142, 143, 144, 145, 148, 149, 140, 141, 142, 143, 144, 145, 148, 149, 146, 147, 148, 149, 278, 146, 147, 148, 149, 279, 144, 145, 146, 147, 148, 149, 152, 153, 144, 145, 146, 147, 148, 149, 152, 153, 150, 151, 152, 153, 285, 150, 151, 152, 153, 286, 148, 149, 150, 151, 152, 153, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 292, 154, 155, 156, 157, 293, 86, 87, 154, 155, 156, 157, 160, 161, 86, 87, 154, 155, 156, 157, 160, 161, 158, 159, 160, 161, 299, 158, 159, 160, 161, 300, 156, 157, 158, 159, 160, 161, 164, 165, 156, 157, 158, 159, 160, 161, 164, 165, 162, 163, 164, 165, 306, 162, 163, 164, 165, 307, 160, 161, 162, 163, 164, 165, 168, 169, 160, 161, 162, 163, 164, 165, 168, 169, 166, 167, 168, 169, 313, 166, 167, 168, 169, 314, 164, 165, 166, 167, 168, 169, 172, 173, 164, 165, 166, 167, 168, 169, 172, 173, 170, 171, 172, 173, 320, 170, 171, 172, 173, 321, 168, 169, 170, 171, 172, 173, 168, 169, 170, 171, 172, 173, 0, 1, 92, 174, 174, 175, 176, 0, 175, 176, 3, 177, 2, 90, 91, 178, 179, 2, 4, 90, 91, 178, 179, 2, 90, 91, 178, 179, 180, 2, 90, 91, 178, 179, 181, 182, 187, 8, 94, 184, 6, 94, 183, 94, 95, 183, 184, 185, 186, 94, 95, 183, 184, 185, 186, 94, 95, 185, 186, 187, 94, 95, 185, 186, 188, 189, 194, 12, 98, 191, 10, 98, 190, 98, 99, 190, 191, 192, 193, 98, 99, 190, 191, 192, 193, 98, 99, 192, 193, 194, 98, 99, 192, 193, 195, 196, 201, 16, 102, 198, 14, 102, 197, 102, 103, 197, 198, 199, 200, 102, 103, 197, 198, 199, 200, 102, 103, 199, 200, 201, 102, 103, 199, 200, 202, 203, 208, 20, 106, 205, 18, 106, 204, 106, 107, 204, 205, 206, 207, 106, 107, 204, 205, 206, 207, 106, 107, 206, 207, 208, 106, 107, 206, 207, 209, 210, 215, 24, 110, 212, 22, 110, 211, 110, 111, 211, 212, 213, 214, 110, 111, 211, 212, 213, 214, 110, 111, 213, 214, 215, 110, 111, 213, 214, 216, 217, 222, 28, 114, 219, 26, 114, 218, 114, 115, 218, 219, 220, 221, 114, 115, 218, 219, 220, 221, 114, 115, 220, 221, 222, 114, 115, 220, 221, 223, 224, 229, 32, 118, 226, 30, 118, 225, 118, 119, 225, 226, 227, 228, 118, 119, 225, 226, 227, 228, 118, 119, 227, 228, 229, 118, 119, 227, 228, 230, 231, 236, 36, 122, 233, 34, 122, 232, 122, 123, 232, 233, 234, 235, 122, 123, 232, 233, 234, 235, 122, 123, 234, 235, 236, 122, 123, 234, 235, 237, 238, 243, 40, 126, 240, 38, 126, 239, 126, 127, 239, 240, 241, 242, 126, 127, 239, 240, 241, 242, 126, 127, 241, 242, 243, 126, 127, 241, 242, 244, 245, 250, 44, 130, 247, 42, 130, 246, 130, 131, 246, 247, 248, 249, 130, 131, 246, 247, 248, 249, 130, 131, 248, 249, 250, 130, 131, 248, 249, 251, 252, 257, 48, 134, 254, 46, 134, 253, 134, 135, 253, 254, 255, 256, 134, 135, 253, 254, 255, 256, 134, 135, 255, 256, 257, 134, 135, 255, 256, 258, 259, 264, 52, 138, 261, 50, 138, 260, 138, 139, 260, 261, 262, 263, 138, 139, 260, 261, 262, 263, 138, 139, 262, 263, 264, 138, 139, 262, 263, 265, 266, 271, 56, 142, 268, 54, 142, 267, 142, 143, 267, 268, 269, 270, 142, 143, 267, 268, 269, 270, 142, 143, 269, 270, 271, 142, 143, 269, 270, 272, 273, 278, 60, 146, 275, 58, 146, 274, 146, 147, 274, 275, 276, 277, 146, 147, 274, 275, 276, 277, 146, 147, 276, 277, 278, 146, 147, 276, 277, 279, 280, 285, 64, 150, 282, 62, 150, 281, 150, 151, 281, 282, 283, 284, 150, 151, 281, 282, 283, 284, 150, 151, 283, 284, 285, 150, 151, 283, 284, 286, 287, 292, 68, 154, 289, 66, 154, 288, 154, 155, 288, 289, 290, 291, 154, 155, 288, 289, 290, 291, 154, 155, 290, 291, 292, 154, 155, 290, 291, 293, 294, 299, 72, 158, 296, 70, 158, 295, 158, 159, 295, 296, 297, 298, 158, 159, 295, 296, 297, 298, 158, 159, 297, 298, 299, 158, 159, 297, 298, 300, 301, 306, 76, 162, 303, 74, 162, 302, 162, 163, 302, 303, 304, 305, 162, 163, 302, 303, 304, 305, 162, 163, 304, 305, 306, 162, 163, 304, 305, 307, 308, 313, 80, 166, 310, 78, 166, 309, 166, 167, 309, 310, 311, 312, 166, 167, 309, 310, 311, 312, 166, 167, 311, 312, 313, 166, 167, 311, 312, 314, 315, 320, 84, 170, 317, 82, 170, 316, 170, 171, 316, 317, 318, 319, 170, 171, 316, 317, 318, 319, 170, 171, 318, 319, 320, 170, 171, 318, 319, 321, 177, 322, 85, 322, 323]
    sp_jac_trap_ja = [0, 2, 4, 7, 8, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33, 34, 36, 37, 39, 40, 42, 43, 45, 46, 48, 49, 51, 52, 54, 55, 57, 58, 60, 61, 63, 64, 66, 67, 69, 70, 72, 73, 75, 76, 78, 79, 81, 82, 84, 85, 87, 88, 90, 91, 93, 94, 96, 97, 99, 100, 102, 103, 105, 106, 108, 109, 111, 112, 114, 115, 117, 118, 120, 121, 123, 124, 126, 127, 129, 131, 145, 159, 165, 171, 176, 181, 187, 191, 196, 201, 209, 217, 222, 227, 235, 243, 248, 253, 261, 269, 274, 279, 287, 295, 300, 305, 311, 317, 322, 327, 335, 343, 348, 353, 361, 369, 374, 379, 387, 395, 400, 405, 413, 421, 426, 431, 437, 443, 448, 453, 461, 469, 474, 479, 487, 495, 500, 505, 513, 521, 526, 531, 539, 547, 552, 557, 563, 569, 574, 579, 587, 595, 600, 605, 613, 621, 626, 631, 639, 647, 652, 657, 665, 673, 678, 683, 689, 695, 699, 702, 705, 707, 712, 718, 724, 730, 732, 735, 738, 744, 750, 755, 760, 762, 765, 768, 774, 780, 785, 790, 792, 795, 798, 804, 810, 815, 820, 822, 825, 828, 834, 840, 845, 850, 852, 855, 858, 864, 870, 875, 880, 882, 885, 888, 894, 900, 905, 910, 912, 915, 918, 924, 930, 935, 940, 942, 945, 948, 954, 960, 965, 970, 972, 975, 978, 984, 990, 995, 1000, 1002, 1005, 1008, 1014, 1020, 1025, 1030, 1032, 1035, 1038, 1044, 1050, 1055, 1060, 1062, 1065, 1068, 1074, 1080, 1085, 1090, 1092, 1095, 1098, 1104, 1110, 1115, 1120, 1122, 1125, 1128, 1134, 1140, 1145, 1150, 1152, 1155, 1158, 1164, 1170, 1175, 1180, 1182, 1185, 1188, 1194, 1200, 1205, 1210, 1212, 1215, 1218, 1224, 1230, 1235, 1240, 1242, 1245, 1248, 1254, 1260, 1265, 1270, 1272, 1275, 1278, 1284, 1290, 1295, 1300, 1302, 1305, 1308, 1314, 1320, 1325, 1330, 1332, 1335]
    sp_jac_trap_nia = 324
    sp_jac_trap_nja = 324
    return sp_jac_trap_ia, sp_jac_trap_ja, sp_jac_trap_nia, sp_jac_trap_nja 
