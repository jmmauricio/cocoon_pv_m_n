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
    import pv_2_20_cffi as jacs
if dae_file_mode == 'enviroment':
    import envus.no_enviroment.pv_2_20_cffi as jacs
if dae_file_mode == 'colab':
    import pv_2_20_cffi as jacs
if dae_file_mode == 'testing':
    from pydae.temp import pv_2_20_cffi as jacs
    
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
        self.N_x = 166
        self.N_y = 458 
        self.N_z = 268 
        self.N_store = 100000 
        self.params_list = ['S_base', 'g_POI_GRID', 'b_POI_GRID', 'bs_POI_GRID', 'g_BESS_POI_MV', 'b_BESS_POI_MV', 'bs_BESS_POI_MV', 'g_LV0101_MV0101', 'b_LV0101_MV0101', 'bs_LV0101_MV0101', 'g_MV0101_POI_MV', 'b_MV0101_POI_MV', 'bs_MV0101_POI_MV', 'g_LV0102_MV0102', 'b_LV0102_MV0102', 'bs_LV0102_MV0102', 'g_MV0102_MV0101', 'b_MV0102_MV0101', 'bs_MV0102_MV0101', 'g_LV0103_MV0103', 'b_LV0103_MV0103', 'bs_LV0103_MV0103', 'g_MV0103_MV0102', 'b_MV0103_MV0102', 'bs_MV0103_MV0102', 'g_LV0104_MV0104', 'b_LV0104_MV0104', 'bs_LV0104_MV0104', 'g_MV0104_MV0103', 'b_MV0104_MV0103', 'bs_MV0104_MV0103', 'g_LV0105_MV0105', 'b_LV0105_MV0105', 'bs_LV0105_MV0105', 'g_MV0105_MV0104', 'b_MV0105_MV0104', 'bs_MV0105_MV0104', 'g_LV0106_MV0106', 'b_LV0106_MV0106', 'bs_LV0106_MV0106', 'g_MV0106_MV0105', 'b_MV0106_MV0105', 'bs_MV0106_MV0105', 'g_LV0107_MV0107', 'b_LV0107_MV0107', 'bs_LV0107_MV0107', 'g_MV0107_MV0106', 'b_MV0107_MV0106', 'bs_MV0107_MV0106', 'g_LV0108_MV0108', 'b_LV0108_MV0108', 'bs_LV0108_MV0108', 'g_MV0108_MV0107', 'b_MV0108_MV0107', 'bs_MV0108_MV0107', 'g_LV0109_MV0109', 'b_LV0109_MV0109', 'bs_LV0109_MV0109', 'g_MV0109_MV0108', 'b_MV0109_MV0108', 'bs_MV0109_MV0108', 'g_LV0110_MV0110', 'b_LV0110_MV0110', 'bs_LV0110_MV0110', 'g_MV0110_MV0109', 'b_MV0110_MV0109', 'bs_MV0110_MV0109', 'g_LV0111_MV0111', 'b_LV0111_MV0111', 'bs_LV0111_MV0111', 'g_MV0111_MV0110', 'b_MV0111_MV0110', 'bs_MV0111_MV0110', 'g_LV0112_MV0112', 'b_LV0112_MV0112', 'bs_LV0112_MV0112', 'g_MV0112_MV0111', 'b_MV0112_MV0111', 'bs_MV0112_MV0111', 'g_LV0113_MV0113', 'b_LV0113_MV0113', 'bs_LV0113_MV0113', 'g_MV0113_MV0112', 'b_MV0113_MV0112', 'bs_MV0113_MV0112', 'g_LV0114_MV0114', 'b_LV0114_MV0114', 'bs_LV0114_MV0114', 'g_MV0114_MV0113', 'b_MV0114_MV0113', 'bs_MV0114_MV0113', 'g_LV0115_MV0115', 'b_LV0115_MV0115', 'bs_LV0115_MV0115', 'g_MV0115_MV0114', 'b_MV0115_MV0114', 'bs_MV0115_MV0114', 'g_LV0116_MV0116', 'b_LV0116_MV0116', 'bs_LV0116_MV0116', 'g_MV0116_MV0115', 'b_MV0116_MV0115', 'bs_MV0116_MV0115', 'g_LV0117_MV0117', 'b_LV0117_MV0117', 'bs_LV0117_MV0117', 'g_MV0117_MV0116', 'b_MV0117_MV0116', 'bs_MV0117_MV0116', 'g_LV0118_MV0118', 'b_LV0118_MV0118', 'bs_LV0118_MV0118', 'g_MV0118_MV0117', 'b_MV0118_MV0117', 'bs_MV0118_MV0117', 'g_LV0119_MV0119', 'b_LV0119_MV0119', 'bs_LV0119_MV0119', 'g_MV0119_MV0118', 'b_MV0119_MV0118', 'bs_MV0119_MV0118', 'g_LV0120_MV0120', 'b_LV0120_MV0120', 'bs_LV0120_MV0120', 'g_MV0120_MV0119', 'b_MV0120_MV0119', 'bs_MV0120_MV0119', 'g_LV0201_MV0201', 'b_LV0201_MV0201', 'bs_LV0201_MV0201', 'g_MV0201_POI_MV', 'b_MV0201_POI_MV', 'bs_MV0201_POI_MV', 'g_LV0202_MV0202', 'b_LV0202_MV0202', 'bs_LV0202_MV0202', 'g_MV0202_MV0201', 'b_MV0202_MV0201', 'bs_MV0202_MV0201', 'g_LV0203_MV0203', 'b_LV0203_MV0203', 'bs_LV0203_MV0203', 'g_MV0203_MV0202', 'b_MV0203_MV0202', 'bs_MV0203_MV0202', 'g_LV0204_MV0204', 'b_LV0204_MV0204', 'bs_LV0204_MV0204', 'g_MV0204_MV0203', 'b_MV0204_MV0203', 'bs_MV0204_MV0203', 'g_LV0205_MV0205', 'b_LV0205_MV0205', 'bs_LV0205_MV0205', 'g_MV0205_MV0204', 'b_MV0205_MV0204', 'bs_MV0205_MV0204', 'g_LV0206_MV0206', 'b_LV0206_MV0206', 'bs_LV0206_MV0206', 'g_MV0206_MV0205', 'b_MV0206_MV0205', 'bs_MV0206_MV0205', 'g_LV0207_MV0207', 'b_LV0207_MV0207', 'bs_LV0207_MV0207', 'g_MV0207_MV0206', 'b_MV0207_MV0206', 'bs_MV0207_MV0206', 'g_LV0208_MV0208', 'b_LV0208_MV0208', 'bs_LV0208_MV0208', 'g_MV0208_MV0207', 'b_MV0208_MV0207', 'bs_MV0208_MV0207', 'g_LV0209_MV0209', 'b_LV0209_MV0209', 'bs_LV0209_MV0209', 'g_MV0209_MV0208', 'b_MV0209_MV0208', 'bs_MV0209_MV0208', 'g_LV0210_MV0210', 'b_LV0210_MV0210', 'bs_LV0210_MV0210', 'g_MV0210_MV0209', 'b_MV0210_MV0209', 'bs_MV0210_MV0209', 'g_LV0211_MV0211', 'b_LV0211_MV0211', 'bs_LV0211_MV0211', 'g_MV0211_MV0210', 'b_MV0211_MV0210', 'bs_MV0211_MV0210', 'g_LV0212_MV0212', 'b_LV0212_MV0212', 'bs_LV0212_MV0212', 'g_MV0212_MV0211', 'b_MV0212_MV0211', 'bs_MV0212_MV0211', 'g_LV0213_MV0213', 'b_LV0213_MV0213', 'bs_LV0213_MV0213', 'g_MV0213_MV0212', 'b_MV0213_MV0212', 'bs_MV0213_MV0212', 'g_LV0214_MV0214', 'b_LV0214_MV0214', 'bs_LV0214_MV0214', 'g_MV0214_MV0213', 'b_MV0214_MV0213', 'bs_MV0214_MV0213', 'g_LV0215_MV0215', 'b_LV0215_MV0215', 'bs_LV0215_MV0215', 'g_MV0215_MV0214', 'b_MV0215_MV0214', 'bs_MV0215_MV0214', 'g_LV0216_MV0216', 'b_LV0216_MV0216', 'bs_LV0216_MV0216', 'g_MV0216_MV0215', 'b_MV0216_MV0215', 'bs_MV0216_MV0215', 'g_LV0217_MV0217', 'b_LV0217_MV0217', 'bs_LV0217_MV0217', 'g_MV0217_MV0216', 'b_MV0217_MV0216', 'bs_MV0217_MV0216', 'g_LV0218_MV0218', 'b_LV0218_MV0218', 'bs_LV0218_MV0218', 'g_MV0218_MV0217', 'b_MV0218_MV0217', 'bs_MV0218_MV0217', 'g_LV0219_MV0219', 'b_LV0219_MV0219', 'bs_LV0219_MV0219', 'g_MV0219_MV0218', 'b_MV0219_MV0218', 'bs_MV0219_MV0218', 'g_LV0220_MV0220', 'b_LV0220_MV0220', 'bs_LV0220_MV0220', 'g_MV0220_MV0219', 'b_MV0220_MV0219', 'bs_MV0220_MV0219', 'g_cc_POI_MV_POI', 'b_cc_POI_MV_POI', 'tap_POI_MV_POI', 'ang_POI_MV_POI', 'U_POI_MV_n', 'U_POI_n', 'U_GRID_n', 'U_BESS_n', 'U_LV0101_n', 'U_MV0101_n', 'U_LV0102_n', 'U_MV0102_n', 'U_LV0103_n', 'U_MV0103_n', 'U_LV0104_n', 'U_MV0104_n', 'U_LV0105_n', 'U_MV0105_n', 'U_LV0106_n', 'U_MV0106_n', 'U_LV0107_n', 'U_MV0107_n', 'U_LV0108_n', 'U_MV0108_n', 'U_LV0109_n', 'U_MV0109_n', 'U_LV0110_n', 'U_MV0110_n', 'U_LV0111_n', 'U_MV0111_n', 'U_LV0112_n', 'U_MV0112_n', 'U_LV0113_n', 'U_MV0113_n', 'U_LV0114_n', 'U_MV0114_n', 'U_LV0115_n', 'U_MV0115_n', 'U_LV0116_n', 'U_MV0116_n', 'U_LV0117_n', 'U_MV0117_n', 'U_LV0118_n', 'U_MV0118_n', 'U_LV0119_n', 'U_MV0119_n', 'U_LV0120_n', 'U_MV0120_n', 'U_LV0201_n', 'U_MV0201_n', 'U_LV0202_n', 'U_MV0202_n', 'U_LV0203_n', 'U_MV0203_n', 'U_LV0204_n', 'U_MV0204_n', 'U_LV0205_n', 'U_MV0205_n', 'U_LV0206_n', 'U_MV0206_n', 'U_LV0207_n', 'U_MV0207_n', 'U_LV0208_n', 'U_MV0208_n', 'U_LV0209_n', 'U_MV0209_n', 'U_LV0210_n', 'U_MV0210_n', 'U_LV0211_n', 'U_MV0211_n', 'U_LV0212_n', 'U_MV0212_n', 'U_LV0213_n', 'U_MV0213_n', 'U_LV0214_n', 'U_MV0214_n', 'U_LV0215_n', 'U_MV0215_n', 'U_LV0216_n', 'U_MV0216_n', 'U_LV0217_n', 'U_MV0217_n', 'U_LV0218_n', 'U_MV0218_n', 'U_LV0219_n', 'U_MV0219_n', 'U_LV0220_n', 'U_MV0220_n', 'K_p_BESS', 'K_i_BESS', 'soc_min_BESS', 'soc_max_BESS', 'S_n_BESS', 'E_kWh_BESS', 'A_loss_BESS', 'B_loss_BESS', 'C_loss_BESS', 'R_bat_BESS', 'S_n_GRID', 'F_n_GRID', 'X_v_GRID', 'R_v_GRID', 'K_delta_GRID', 'K_alpha_GRID', 'K_rocov_GRID', 'I_sc_LV0101', 'I_mp_LV0101', 'V_mp_LV0101', 'V_oc_LV0101', 'N_pv_s_LV0101', 'N_pv_p_LV0101', 'K_vt_LV0101', 'K_it_LV0101', 'v_lvrt_LV0101', 'T_lp1p_LV0101', 'T_lp2p_LV0101', 'T_lp1q_LV0101', 'T_lp2q_LV0101', 'PRampUp_LV0101', 'PRampDown_LV0101', 'QRampUp_LV0101', 'QRampDown_LV0101', 'S_n_LV0101', 'F_n_LV0101', 'U_n_LV0101', 'X_s_LV0101', 'R_s_LV0101', 'I_sc_LV0102', 'I_mp_LV0102', 'V_mp_LV0102', 'V_oc_LV0102', 'N_pv_s_LV0102', 'N_pv_p_LV0102', 'K_vt_LV0102', 'K_it_LV0102', 'v_lvrt_LV0102', 'T_lp1p_LV0102', 'T_lp2p_LV0102', 'T_lp1q_LV0102', 'T_lp2q_LV0102', 'PRampUp_LV0102', 'PRampDown_LV0102', 'QRampUp_LV0102', 'QRampDown_LV0102', 'S_n_LV0102', 'F_n_LV0102', 'U_n_LV0102', 'X_s_LV0102', 'R_s_LV0102', 'I_sc_LV0103', 'I_mp_LV0103', 'V_mp_LV0103', 'V_oc_LV0103', 'N_pv_s_LV0103', 'N_pv_p_LV0103', 'K_vt_LV0103', 'K_it_LV0103', 'v_lvrt_LV0103', 'T_lp1p_LV0103', 'T_lp2p_LV0103', 'T_lp1q_LV0103', 'T_lp2q_LV0103', 'PRampUp_LV0103', 'PRampDown_LV0103', 'QRampUp_LV0103', 'QRampDown_LV0103', 'S_n_LV0103', 'F_n_LV0103', 'U_n_LV0103', 'X_s_LV0103', 'R_s_LV0103', 'I_sc_LV0104', 'I_mp_LV0104', 'V_mp_LV0104', 'V_oc_LV0104', 'N_pv_s_LV0104', 'N_pv_p_LV0104', 'K_vt_LV0104', 'K_it_LV0104', 'v_lvrt_LV0104', 'T_lp1p_LV0104', 'T_lp2p_LV0104', 'T_lp1q_LV0104', 'T_lp2q_LV0104', 'PRampUp_LV0104', 'PRampDown_LV0104', 'QRampUp_LV0104', 'QRampDown_LV0104', 'S_n_LV0104', 'F_n_LV0104', 'U_n_LV0104', 'X_s_LV0104', 'R_s_LV0104', 'I_sc_LV0105', 'I_mp_LV0105', 'V_mp_LV0105', 'V_oc_LV0105', 'N_pv_s_LV0105', 'N_pv_p_LV0105', 'K_vt_LV0105', 'K_it_LV0105', 'v_lvrt_LV0105', 'T_lp1p_LV0105', 'T_lp2p_LV0105', 'T_lp1q_LV0105', 'T_lp2q_LV0105', 'PRampUp_LV0105', 'PRampDown_LV0105', 'QRampUp_LV0105', 'QRampDown_LV0105', 'S_n_LV0105', 'F_n_LV0105', 'U_n_LV0105', 'X_s_LV0105', 'R_s_LV0105', 'I_sc_LV0106', 'I_mp_LV0106', 'V_mp_LV0106', 'V_oc_LV0106', 'N_pv_s_LV0106', 'N_pv_p_LV0106', 'K_vt_LV0106', 'K_it_LV0106', 'v_lvrt_LV0106', 'T_lp1p_LV0106', 'T_lp2p_LV0106', 'T_lp1q_LV0106', 'T_lp2q_LV0106', 'PRampUp_LV0106', 'PRampDown_LV0106', 'QRampUp_LV0106', 'QRampDown_LV0106', 'S_n_LV0106', 'F_n_LV0106', 'U_n_LV0106', 'X_s_LV0106', 'R_s_LV0106', 'I_sc_LV0107', 'I_mp_LV0107', 'V_mp_LV0107', 'V_oc_LV0107', 'N_pv_s_LV0107', 'N_pv_p_LV0107', 'K_vt_LV0107', 'K_it_LV0107', 'v_lvrt_LV0107', 'T_lp1p_LV0107', 'T_lp2p_LV0107', 'T_lp1q_LV0107', 'T_lp2q_LV0107', 'PRampUp_LV0107', 'PRampDown_LV0107', 'QRampUp_LV0107', 'QRampDown_LV0107', 'S_n_LV0107', 'F_n_LV0107', 'U_n_LV0107', 'X_s_LV0107', 'R_s_LV0107', 'I_sc_LV0108', 'I_mp_LV0108', 'V_mp_LV0108', 'V_oc_LV0108', 'N_pv_s_LV0108', 'N_pv_p_LV0108', 'K_vt_LV0108', 'K_it_LV0108', 'v_lvrt_LV0108', 'T_lp1p_LV0108', 'T_lp2p_LV0108', 'T_lp1q_LV0108', 'T_lp2q_LV0108', 'PRampUp_LV0108', 'PRampDown_LV0108', 'QRampUp_LV0108', 'QRampDown_LV0108', 'S_n_LV0108', 'F_n_LV0108', 'U_n_LV0108', 'X_s_LV0108', 'R_s_LV0108', 'I_sc_LV0109', 'I_mp_LV0109', 'V_mp_LV0109', 'V_oc_LV0109', 'N_pv_s_LV0109', 'N_pv_p_LV0109', 'K_vt_LV0109', 'K_it_LV0109', 'v_lvrt_LV0109', 'T_lp1p_LV0109', 'T_lp2p_LV0109', 'T_lp1q_LV0109', 'T_lp2q_LV0109', 'PRampUp_LV0109', 'PRampDown_LV0109', 'QRampUp_LV0109', 'QRampDown_LV0109', 'S_n_LV0109', 'F_n_LV0109', 'U_n_LV0109', 'X_s_LV0109', 'R_s_LV0109', 'I_sc_LV0110', 'I_mp_LV0110', 'V_mp_LV0110', 'V_oc_LV0110', 'N_pv_s_LV0110', 'N_pv_p_LV0110', 'K_vt_LV0110', 'K_it_LV0110', 'v_lvrt_LV0110', 'T_lp1p_LV0110', 'T_lp2p_LV0110', 'T_lp1q_LV0110', 'T_lp2q_LV0110', 'PRampUp_LV0110', 'PRampDown_LV0110', 'QRampUp_LV0110', 'QRampDown_LV0110', 'S_n_LV0110', 'F_n_LV0110', 'U_n_LV0110', 'X_s_LV0110', 'R_s_LV0110', 'I_sc_LV0111', 'I_mp_LV0111', 'V_mp_LV0111', 'V_oc_LV0111', 'N_pv_s_LV0111', 'N_pv_p_LV0111', 'K_vt_LV0111', 'K_it_LV0111', 'v_lvrt_LV0111', 'T_lp1p_LV0111', 'T_lp2p_LV0111', 'T_lp1q_LV0111', 'T_lp2q_LV0111', 'PRampUp_LV0111', 'PRampDown_LV0111', 'QRampUp_LV0111', 'QRampDown_LV0111', 'S_n_LV0111', 'F_n_LV0111', 'U_n_LV0111', 'X_s_LV0111', 'R_s_LV0111', 'I_sc_LV0112', 'I_mp_LV0112', 'V_mp_LV0112', 'V_oc_LV0112', 'N_pv_s_LV0112', 'N_pv_p_LV0112', 'K_vt_LV0112', 'K_it_LV0112', 'v_lvrt_LV0112', 'T_lp1p_LV0112', 'T_lp2p_LV0112', 'T_lp1q_LV0112', 'T_lp2q_LV0112', 'PRampUp_LV0112', 'PRampDown_LV0112', 'QRampUp_LV0112', 'QRampDown_LV0112', 'S_n_LV0112', 'F_n_LV0112', 'U_n_LV0112', 'X_s_LV0112', 'R_s_LV0112', 'I_sc_LV0113', 'I_mp_LV0113', 'V_mp_LV0113', 'V_oc_LV0113', 'N_pv_s_LV0113', 'N_pv_p_LV0113', 'K_vt_LV0113', 'K_it_LV0113', 'v_lvrt_LV0113', 'T_lp1p_LV0113', 'T_lp2p_LV0113', 'T_lp1q_LV0113', 'T_lp2q_LV0113', 'PRampUp_LV0113', 'PRampDown_LV0113', 'QRampUp_LV0113', 'QRampDown_LV0113', 'S_n_LV0113', 'F_n_LV0113', 'U_n_LV0113', 'X_s_LV0113', 'R_s_LV0113', 'I_sc_LV0114', 'I_mp_LV0114', 'V_mp_LV0114', 'V_oc_LV0114', 'N_pv_s_LV0114', 'N_pv_p_LV0114', 'K_vt_LV0114', 'K_it_LV0114', 'v_lvrt_LV0114', 'T_lp1p_LV0114', 'T_lp2p_LV0114', 'T_lp1q_LV0114', 'T_lp2q_LV0114', 'PRampUp_LV0114', 'PRampDown_LV0114', 'QRampUp_LV0114', 'QRampDown_LV0114', 'S_n_LV0114', 'F_n_LV0114', 'U_n_LV0114', 'X_s_LV0114', 'R_s_LV0114', 'I_sc_LV0115', 'I_mp_LV0115', 'V_mp_LV0115', 'V_oc_LV0115', 'N_pv_s_LV0115', 'N_pv_p_LV0115', 'K_vt_LV0115', 'K_it_LV0115', 'v_lvrt_LV0115', 'T_lp1p_LV0115', 'T_lp2p_LV0115', 'T_lp1q_LV0115', 'T_lp2q_LV0115', 'PRampUp_LV0115', 'PRampDown_LV0115', 'QRampUp_LV0115', 'QRampDown_LV0115', 'S_n_LV0115', 'F_n_LV0115', 'U_n_LV0115', 'X_s_LV0115', 'R_s_LV0115', 'I_sc_LV0116', 'I_mp_LV0116', 'V_mp_LV0116', 'V_oc_LV0116', 'N_pv_s_LV0116', 'N_pv_p_LV0116', 'K_vt_LV0116', 'K_it_LV0116', 'v_lvrt_LV0116', 'T_lp1p_LV0116', 'T_lp2p_LV0116', 'T_lp1q_LV0116', 'T_lp2q_LV0116', 'PRampUp_LV0116', 'PRampDown_LV0116', 'QRampUp_LV0116', 'QRampDown_LV0116', 'S_n_LV0116', 'F_n_LV0116', 'U_n_LV0116', 'X_s_LV0116', 'R_s_LV0116', 'I_sc_LV0117', 'I_mp_LV0117', 'V_mp_LV0117', 'V_oc_LV0117', 'N_pv_s_LV0117', 'N_pv_p_LV0117', 'K_vt_LV0117', 'K_it_LV0117', 'v_lvrt_LV0117', 'T_lp1p_LV0117', 'T_lp2p_LV0117', 'T_lp1q_LV0117', 'T_lp2q_LV0117', 'PRampUp_LV0117', 'PRampDown_LV0117', 'QRampUp_LV0117', 'QRampDown_LV0117', 'S_n_LV0117', 'F_n_LV0117', 'U_n_LV0117', 'X_s_LV0117', 'R_s_LV0117', 'I_sc_LV0118', 'I_mp_LV0118', 'V_mp_LV0118', 'V_oc_LV0118', 'N_pv_s_LV0118', 'N_pv_p_LV0118', 'K_vt_LV0118', 'K_it_LV0118', 'v_lvrt_LV0118', 'T_lp1p_LV0118', 'T_lp2p_LV0118', 'T_lp1q_LV0118', 'T_lp2q_LV0118', 'PRampUp_LV0118', 'PRampDown_LV0118', 'QRampUp_LV0118', 'QRampDown_LV0118', 'S_n_LV0118', 'F_n_LV0118', 'U_n_LV0118', 'X_s_LV0118', 'R_s_LV0118', 'I_sc_LV0119', 'I_mp_LV0119', 'V_mp_LV0119', 'V_oc_LV0119', 'N_pv_s_LV0119', 'N_pv_p_LV0119', 'K_vt_LV0119', 'K_it_LV0119', 'v_lvrt_LV0119', 'T_lp1p_LV0119', 'T_lp2p_LV0119', 'T_lp1q_LV0119', 'T_lp2q_LV0119', 'PRampUp_LV0119', 'PRampDown_LV0119', 'QRampUp_LV0119', 'QRampDown_LV0119', 'S_n_LV0119', 'F_n_LV0119', 'U_n_LV0119', 'X_s_LV0119', 'R_s_LV0119', 'I_sc_LV0120', 'I_mp_LV0120', 'V_mp_LV0120', 'V_oc_LV0120', 'N_pv_s_LV0120', 'N_pv_p_LV0120', 'K_vt_LV0120', 'K_it_LV0120', 'v_lvrt_LV0120', 'T_lp1p_LV0120', 'T_lp2p_LV0120', 'T_lp1q_LV0120', 'T_lp2q_LV0120', 'PRampUp_LV0120', 'PRampDown_LV0120', 'QRampUp_LV0120', 'QRampDown_LV0120', 'S_n_LV0120', 'F_n_LV0120', 'U_n_LV0120', 'X_s_LV0120', 'R_s_LV0120', 'I_sc_LV0201', 'I_mp_LV0201', 'V_mp_LV0201', 'V_oc_LV0201', 'N_pv_s_LV0201', 'N_pv_p_LV0201', 'K_vt_LV0201', 'K_it_LV0201', 'v_lvrt_LV0201', 'T_lp1p_LV0201', 'T_lp2p_LV0201', 'T_lp1q_LV0201', 'T_lp2q_LV0201', 'PRampUp_LV0201', 'PRampDown_LV0201', 'QRampUp_LV0201', 'QRampDown_LV0201', 'S_n_LV0201', 'F_n_LV0201', 'U_n_LV0201', 'X_s_LV0201', 'R_s_LV0201', 'I_sc_LV0202', 'I_mp_LV0202', 'V_mp_LV0202', 'V_oc_LV0202', 'N_pv_s_LV0202', 'N_pv_p_LV0202', 'K_vt_LV0202', 'K_it_LV0202', 'v_lvrt_LV0202', 'T_lp1p_LV0202', 'T_lp2p_LV0202', 'T_lp1q_LV0202', 'T_lp2q_LV0202', 'PRampUp_LV0202', 'PRampDown_LV0202', 'QRampUp_LV0202', 'QRampDown_LV0202', 'S_n_LV0202', 'F_n_LV0202', 'U_n_LV0202', 'X_s_LV0202', 'R_s_LV0202', 'I_sc_LV0203', 'I_mp_LV0203', 'V_mp_LV0203', 'V_oc_LV0203', 'N_pv_s_LV0203', 'N_pv_p_LV0203', 'K_vt_LV0203', 'K_it_LV0203', 'v_lvrt_LV0203', 'T_lp1p_LV0203', 'T_lp2p_LV0203', 'T_lp1q_LV0203', 'T_lp2q_LV0203', 'PRampUp_LV0203', 'PRampDown_LV0203', 'QRampUp_LV0203', 'QRampDown_LV0203', 'S_n_LV0203', 'F_n_LV0203', 'U_n_LV0203', 'X_s_LV0203', 'R_s_LV0203', 'I_sc_LV0204', 'I_mp_LV0204', 'V_mp_LV0204', 'V_oc_LV0204', 'N_pv_s_LV0204', 'N_pv_p_LV0204', 'K_vt_LV0204', 'K_it_LV0204', 'v_lvrt_LV0204', 'T_lp1p_LV0204', 'T_lp2p_LV0204', 'T_lp1q_LV0204', 'T_lp2q_LV0204', 'PRampUp_LV0204', 'PRampDown_LV0204', 'QRampUp_LV0204', 'QRampDown_LV0204', 'S_n_LV0204', 'F_n_LV0204', 'U_n_LV0204', 'X_s_LV0204', 'R_s_LV0204', 'I_sc_LV0205', 'I_mp_LV0205', 'V_mp_LV0205', 'V_oc_LV0205', 'N_pv_s_LV0205', 'N_pv_p_LV0205', 'K_vt_LV0205', 'K_it_LV0205', 'v_lvrt_LV0205', 'T_lp1p_LV0205', 'T_lp2p_LV0205', 'T_lp1q_LV0205', 'T_lp2q_LV0205', 'PRampUp_LV0205', 'PRampDown_LV0205', 'QRampUp_LV0205', 'QRampDown_LV0205', 'S_n_LV0205', 'F_n_LV0205', 'U_n_LV0205', 'X_s_LV0205', 'R_s_LV0205', 'I_sc_LV0206', 'I_mp_LV0206', 'V_mp_LV0206', 'V_oc_LV0206', 'N_pv_s_LV0206', 'N_pv_p_LV0206', 'K_vt_LV0206', 'K_it_LV0206', 'v_lvrt_LV0206', 'T_lp1p_LV0206', 'T_lp2p_LV0206', 'T_lp1q_LV0206', 'T_lp2q_LV0206', 'PRampUp_LV0206', 'PRampDown_LV0206', 'QRampUp_LV0206', 'QRampDown_LV0206', 'S_n_LV0206', 'F_n_LV0206', 'U_n_LV0206', 'X_s_LV0206', 'R_s_LV0206', 'I_sc_LV0207', 'I_mp_LV0207', 'V_mp_LV0207', 'V_oc_LV0207', 'N_pv_s_LV0207', 'N_pv_p_LV0207', 'K_vt_LV0207', 'K_it_LV0207', 'v_lvrt_LV0207', 'T_lp1p_LV0207', 'T_lp2p_LV0207', 'T_lp1q_LV0207', 'T_lp2q_LV0207', 'PRampUp_LV0207', 'PRampDown_LV0207', 'QRampUp_LV0207', 'QRampDown_LV0207', 'S_n_LV0207', 'F_n_LV0207', 'U_n_LV0207', 'X_s_LV0207', 'R_s_LV0207', 'I_sc_LV0208', 'I_mp_LV0208', 'V_mp_LV0208', 'V_oc_LV0208', 'N_pv_s_LV0208', 'N_pv_p_LV0208', 'K_vt_LV0208', 'K_it_LV0208', 'v_lvrt_LV0208', 'T_lp1p_LV0208', 'T_lp2p_LV0208', 'T_lp1q_LV0208', 'T_lp2q_LV0208', 'PRampUp_LV0208', 'PRampDown_LV0208', 'QRampUp_LV0208', 'QRampDown_LV0208', 'S_n_LV0208', 'F_n_LV0208', 'U_n_LV0208', 'X_s_LV0208', 'R_s_LV0208', 'I_sc_LV0209', 'I_mp_LV0209', 'V_mp_LV0209', 'V_oc_LV0209', 'N_pv_s_LV0209', 'N_pv_p_LV0209', 'K_vt_LV0209', 'K_it_LV0209', 'v_lvrt_LV0209', 'T_lp1p_LV0209', 'T_lp2p_LV0209', 'T_lp1q_LV0209', 'T_lp2q_LV0209', 'PRampUp_LV0209', 'PRampDown_LV0209', 'QRampUp_LV0209', 'QRampDown_LV0209', 'S_n_LV0209', 'F_n_LV0209', 'U_n_LV0209', 'X_s_LV0209', 'R_s_LV0209', 'I_sc_LV0210', 'I_mp_LV0210', 'V_mp_LV0210', 'V_oc_LV0210', 'N_pv_s_LV0210', 'N_pv_p_LV0210', 'K_vt_LV0210', 'K_it_LV0210', 'v_lvrt_LV0210', 'T_lp1p_LV0210', 'T_lp2p_LV0210', 'T_lp1q_LV0210', 'T_lp2q_LV0210', 'PRampUp_LV0210', 'PRampDown_LV0210', 'QRampUp_LV0210', 'QRampDown_LV0210', 'S_n_LV0210', 'F_n_LV0210', 'U_n_LV0210', 'X_s_LV0210', 'R_s_LV0210', 'I_sc_LV0211', 'I_mp_LV0211', 'V_mp_LV0211', 'V_oc_LV0211', 'N_pv_s_LV0211', 'N_pv_p_LV0211', 'K_vt_LV0211', 'K_it_LV0211', 'v_lvrt_LV0211', 'T_lp1p_LV0211', 'T_lp2p_LV0211', 'T_lp1q_LV0211', 'T_lp2q_LV0211', 'PRampUp_LV0211', 'PRampDown_LV0211', 'QRampUp_LV0211', 'QRampDown_LV0211', 'S_n_LV0211', 'F_n_LV0211', 'U_n_LV0211', 'X_s_LV0211', 'R_s_LV0211', 'I_sc_LV0212', 'I_mp_LV0212', 'V_mp_LV0212', 'V_oc_LV0212', 'N_pv_s_LV0212', 'N_pv_p_LV0212', 'K_vt_LV0212', 'K_it_LV0212', 'v_lvrt_LV0212', 'T_lp1p_LV0212', 'T_lp2p_LV0212', 'T_lp1q_LV0212', 'T_lp2q_LV0212', 'PRampUp_LV0212', 'PRampDown_LV0212', 'QRampUp_LV0212', 'QRampDown_LV0212', 'S_n_LV0212', 'F_n_LV0212', 'U_n_LV0212', 'X_s_LV0212', 'R_s_LV0212', 'I_sc_LV0213', 'I_mp_LV0213', 'V_mp_LV0213', 'V_oc_LV0213', 'N_pv_s_LV0213', 'N_pv_p_LV0213', 'K_vt_LV0213', 'K_it_LV0213', 'v_lvrt_LV0213', 'T_lp1p_LV0213', 'T_lp2p_LV0213', 'T_lp1q_LV0213', 'T_lp2q_LV0213', 'PRampUp_LV0213', 'PRampDown_LV0213', 'QRampUp_LV0213', 'QRampDown_LV0213', 'S_n_LV0213', 'F_n_LV0213', 'U_n_LV0213', 'X_s_LV0213', 'R_s_LV0213', 'I_sc_LV0214', 'I_mp_LV0214', 'V_mp_LV0214', 'V_oc_LV0214', 'N_pv_s_LV0214', 'N_pv_p_LV0214', 'K_vt_LV0214', 'K_it_LV0214', 'v_lvrt_LV0214', 'T_lp1p_LV0214', 'T_lp2p_LV0214', 'T_lp1q_LV0214', 'T_lp2q_LV0214', 'PRampUp_LV0214', 'PRampDown_LV0214', 'QRampUp_LV0214', 'QRampDown_LV0214', 'S_n_LV0214', 'F_n_LV0214', 'U_n_LV0214', 'X_s_LV0214', 'R_s_LV0214', 'I_sc_LV0215', 'I_mp_LV0215', 'V_mp_LV0215', 'V_oc_LV0215', 'N_pv_s_LV0215', 'N_pv_p_LV0215', 'K_vt_LV0215', 'K_it_LV0215', 'v_lvrt_LV0215', 'T_lp1p_LV0215', 'T_lp2p_LV0215', 'T_lp1q_LV0215', 'T_lp2q_LV0215', 'PRampUp_LV0215', 'PRampDown_LV0215', 'QRampUp_LV0215', 'QRampDown_LV0215', 'S_n_LV0215', 'F_n_LV0215', 'U_n_LV0215', 'X_s_LV0215', 'R_s_LV0215', 'I_sc_LV0216', 'I_mp_LV0216', 'V_mp_LV0216', 'V_oc_LV0216', 'N_pv_s_LV0216', 'N_pv_p_LV0216', 'K_vt_LV0216', 'K_it_LV0216', 'v_lvrt_LV0216', 'T_lp1p_LV0216', 'T_lp2p_LV0216', 'T_lp1q_LV0216', 'T_lp2q_LV0216', 'PRampUp_LV0216', 'PRampDown_LV0216', 'QRampUp_LV0216', 'QRampDown_LV0216', 'S_n_LV0216', 'F_n_LV0216', 'U_n_LV0216', 'X_s_LV0216', 'R_s_LV0216', 'I_sc_LV0217', 'I_mp_LV0217', 'V_mp_LV0217', 'V_oc_LV0217', 'N_pv_s_LV0217', 'N_pv_p_LV0217', 'K_vt_LV0217', 'K_it_LV0217', 'v_lvrt_LV0217', 'T_lp1p_LV0217', 'T_lp2p_LV0217', 'T_lp1q_LV0217', 'T_lp2q_LV0217', 'PRampUp_LV0217', 'PRampDown_LV0217', 'QRampUp_LV0217', 'QRampDown_LV0217', 'S_n_LV0217', 'F_n_LV0217', 'U_n_LV0217', 'X_s_LV0217', 'R_s_LV0217', 'I_sc_LV0218', 'I_mp_LV0218', 'V_mp_LV0218', 'V_oc_LV0218', 'N_pv_s_LV0218', 'N_pv_p_LV0218', 'K_vt_LV0218', 'K_it_LV0218', 'v_lvrt_LV0218', 'T_lp1p_LV0218', 'T_lp2p_LV0218', 'T_lp1q_LV0218', 'T_lp2q_LV0218', 'PRampUp_LV0218', 'PRampDown_LV0218', 'QRampUp_LV0218', 'QRampDown_LV0218', 'S_n_LV0218', 'F_n_LV0218', 'U_n_LV0218', 'X_s_LV0218', 'R_s_LV0218', 'I_sc_LV0219', 'I_mp_LV0219', 'V_mp_LV0219', 'V_oc_LV0219', 'N_pv_s_LV0219', 'N_pv_p_LV0219', 'K_vt_LV0219', 'K_it_LV0219', 'v_lvrt_LV0219', 'T_lp1p_LV0219', 'T_lp2p_LV0219', 'T_lp1q_LV0219', 'T_lp2q_LV0219', 'PRampUp_LV0219', 'PRampDown_LV0219', 'QRampUp_LV0219', 'QRampDown_LV0219', 'S_n_LV0219', 'F_n_LV0219', 'U_n_LV0219', 'X_s_LV0219', 'R_s_LV0219', 'I_sc_LV0220', 'I_mp_LV0220', 'V_mp_LV0220', 'V_oc_LV0220', 'N_pv_s_LV0220', 'N_pv_p_LV0220', 'K_vt_LV0220', 'K_it_LV0220', 'v_lvrt_LV0220', 'T_lp1p_LV0220', 'T_lp2p_LV0220', 'T_lp1q_LV0220', 'T_lp2q_LV0220', 'PRampUp_LV0220', 'PRampDown_LV0220', 'QRampUp_LV0220', 'QRampDown_LV0220', 'S_n_LV0220', 'F_n_LV0220', 'U_n_LV0220', 'X_s_LV0220', 'R_s_LV0220', 'K_p_agc', 'K_i_agc', 'K_xif'] 
        self.params_values_list  = [100000000.0, 0.0, -4.12, 0.0, 0.0, -0.23999999999999996, 0.0, 0.04615384615384615, -0.23076923076923075, 0.0, 12.0, -12.0, 0.0024, 0.04615384615384615, -0.23076923076923075, 0.0, 11.4, -11.4, 0.00228, 0.04615384615384615, -0.23076923076923075, 0.0, 10.799999999999997, -10.799999999999997, 0.0021599999999999996, 0.04615384615384615, -0.23076923076923075, 0.0, 10.200000000000001, -10.200000000000001, 0.00204, 0.04615384615384615, -0.23076923076923075, 0.0, 9.6, -9.6, 0.00192, 0.04615384615384615, -0.23076923076923075, 0.0, 9.0, -9.0, 0.0018, 0.04615384615384615, -0.23076923076923075, 0.0, 8.4, -8.4, 0.00168, 0.04615384615384615, -0.23076923076923075, 0.0, 7.800000000000002, -7.800000000000002, 0.00156, 0.04615384615384615, -0.23076923076923075, 0.0, 7.199999999999999, -7.199999999999999, 0.0014399999999999997, 0.04615384615384615, -0.23076923076923075, 0.0, 6.6, -6.6, 0.00132, 0.04615384615384615, -0.23076923076923075, 0.0, 6.0, -6.0, 0.0012, 0.04615384615384615, -0.23076923076923075, 0.0, 5.399999999999999, -5.399999999999999, 0.0010799999999999998, 0.04615384615384615, -0.23076923076923075, 0.0, 4.8, -4.8, 0.00096, 0.04615384615384615, -0.23076923076923075, 0.0, 4.2, -4.2, 0.00084, 0.04615384615384615, -0.23076923076923075, 0.0, 3.5999999999999996, -3.5999999999999996, 0.0007199999999999998, 0.04615384615384615, -0.23076923076923075, 0.0, 3.0, -3.0, 0.0006, 0.04615384615384615, -0.23076923076923075, 0.0, 2.4, -2.4, 0.00048, 0.04615384615384615, -0.23076923076923075, 0.0, 1.7999999999999998, -1.7999999999999998, 0.0003599999999999999, 0.04615384615384615, -0.23076923076923075, 0.0, 1.2, -1.2, 0.00024, 0.04615384615384615, -0.23076923076923075, 0.0, 0.6, -0.6, 0.00012, 0.04615384615384615, -0.23076923076923075, 0.0, 12.0, -12.0, 0.0024, 0.04615384615384615, -0.23076923076923075, 0.0, 11.4, -11.4, 0.00228, 0.04615384615384615, -0.23076923076923075, 0.0, 10.799999999999997, -10.799999999999997, 0.0021599999999999996, 0.04615384615384615, -0.23076923076923075, 0.0, 10.200000000000001, -10.200000000000001, 0.00204, 0.04615384615384615, -0.23076923076923075, 0.0, 9.6, -9.6, 0.00192, 0.04615384615384615, -0.23076923076923075, 0.0, 9.0, -9.0, 0.0018, 0.04615384615384615, -0.23076923076923075, 0.0, 8.4, -8.4, 0.00168, 0.04615384615384615, -0.23076923076923075, 0.0, 7.800000000000002, -7.800000000000002, 0.00156, 0.04615384615384615, -0.23076923076923075, 0.0, 7.199999999999999, -7.199999999999999, 0.0014399999999999997, 0.04615384615384615, -0.23076923076923075, 0.0, 6.6, -6.6, 0.00132, 0.04615384615384615, -0.23076923076923075, 0.0, 6.0, -6.0, 0.0012, 0.04615384615384615, -0.23076923076923075, 0.0, 5.399999999999999, -5.399999999999999, 0.0010799999999999998, 0.04615384615384615, -0.23076923076923075, 0.0, 4.8, -4.8, 0.00096, 0.04615384615384615, -0.23076923076923075, 0.0, 4.2, -4.2, 0.00084, 0.04615384615384615, -0.23076923076923075, 0.0, 3.5999999999999996, -3.5999999999999996, 0.0007199999999999998, 0.04615384615384615, -0.23076923076923075, 0.0, 3.0, -3.0, 0.0006, 0.04615384615384615, -0.23076923076923075, 0.0, 2.4, -2.4, 0.00048, 0.04615384615384615, -0.23076923076923075, 0.0, 1.7999999999999998, -1.7999999999999998, 0.0003599999999999999, 0.04615384615384615, -0.23076923076923075, 0.0, 1.2, -1.2, 0.00024, 0.04615384615384615, -0.23076923076923075, 0.0, 0.6, -0.6, 0.00012, 0.0, -8.0, 1.0, 0.0, 20000.0, 132000, 132000, 690.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 1e-06, 1e-06, 0.0, 1.0, 1000000.0, 250, 0.0001, 0.0, 0.0001, 0.0, 1000000000.0, 50.0, 0.001, 0.0, 0.001, 1e-06, 1e-06, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 0.1, 0.1, 0.1, 0.1, 2.5, -2.5, 2.5, -2.5, 1000000.0, 50.0, 400.0, 0.1, 0.0001, 0.0, 0.0, 0.01] 
        self.inputs_ini_list = ['P_POI_MV', 'Q_POI_MV', 'P_POI', 'Q_POI', 'P_GRID', 'Q_GRID', 'P_BESS', 'Q_BESS', 'P_LV0101', 'Q_LV0101', 'P_MV0101', 'Q_MV0101', 'P_LV0102', 'Q_LV0102', 'P_MV0102', 'Q_MV0102', 'P_LV0103', 'Q_LV0103', 'P_MV0103', 'Q_MV0103', 'P_LV0104', 'Q_LV0104', 'P_MV0104', 'Q_MV0104', 'P_LV0105', 'Q_LV0105', 'P_MV0105', 'Q_MV0105', 'P_LV0106', 'Q_LV0106', 'P_MV0106', 'Q_MV0106', 'P_LV0107', 'Q_LV0107', 'P_MV0107', 'Q_MV0107', 'P_LV0108', 'Q_LV0108', 'P_MV0108', 'Q_MV0108', 'P_LV0109', 'Q_LV0109', 'P_MV0109', 'Q_MV0109', 'P_LV0110', 'Q_LV0110', 'P_MV0110', 'Q_MV0110', 'P_LV0111', 'Q_LV0111', 'P_MV0111', 'Q_MV0111', 'P_LV0112', 'Q_LV0112', 'P_MV0112', 'Q_MV0112', 'P_LV0113', 'Q_LV0113', 'P_MV0113', 'Q_MV0113', 'P_LV0114', 'Q_LV0114', 'P_MV0114', 'Q_MV0114', 'P_LV0115', 'Q_LV0115', 'P_MV0115', 'Q_MV0115', 'P_LV0116', 'Q_LV0116', 'P_MV0116', 'Q_MV0116', 'P_LV0117', 'Q_LV0117', 'P_MV0117', 'Q_MV0117', 'P_LV0118', 'Q_LV0118', 'P_MV0118', 'Q_MV0118', 'P_LV0119', 'Q_LV0119', 'P_MV0119', 'Q_MV0119', 'P_LV0120', 'Q_LV0120', 'P_MV0120', 'Q_MV0120', 'P_LV0201', 'Q_LV0201', 'P_MV0201', 'Q_MV0201', 'P_LV0202', 'Q_LV0202', 'P_MV0202', 'Q_MV0202', 'P_LV0203', 'Q_LV0203', 'P_MV0203', 'Q_MV0203', 'P_LV0204', 'Q_LV0204', 'P_MV0204', 'Q_MV0204', 'P_LV0205', 'Q_LV0205', 'P_MV0205', 'Q_MV0205', 'P_LV0206', 'Q_LV0206', 'P_MV0206', 'Q_MV0206', 'P_LV0207', 'Q_LV0207', 'P_MV0207', 'Q_MV0207', 'P_LV0208', 'Q_LV0208', 'P_MV0208', 'Q_MV0208', 'P_LV0209', 'Q_LV0209', 'P_MV0209', 'Q_MV0209', 'P_LV0210', 'Q_LV0210', 'P_MV0210', 'Q_MV0210', 'P_LV0211', 'Q_LV0211', 'P_MV0211', 'Q_MV0211', 'P_LV0212', 'Q_LV0212', 'P_MV0212', 'Q_MV0212', 'P_LV0213', 'Q_LV0213', 'P_MV0213', 'Q_MV0213', 'P_LV0214', 'Q_LV0214', 'P_MV0214', 'Q_MV0214', 'P_LV0215', 'Q_LV0215', 'P_MV0215', 'Q_MV0215', 'P_LV0216', 'Q_LV0216', 'P_MV0216', 'Q_MV0216', 'P_LV0217', 'Q_LV0217', 'P_MV0217', 'Q_MV0217', 'P_LV0218', 'Q_LV0218', 'P_MV0218', 'Q_MV0218', 'P_LV0219', 'Q_LV0219', 'P_MV0219', 'Q_MV0219', 'P_LV0220', 'Q_LV0220', 'P_MV0220', 'Q_MV0220', 'p_s_ref_BESS', 'q_s_ref_BESS', 'soc_ref_BESS', 'alpha_GRID', 'v_ref_GRID', 'omega_ref_GRID', 'delta_ref_GRID', 'phi_GRID', 'rocov_GRID', 'irrad_LV0101', 'temp_deg_LV0101', 'lvrt_ext_LV0101', 'ramp_enable_LV0101', 'p_s_ppc_LV0101', 'q_s_ppc_LV0101', 'i_sa_ref_LV0101', 'i_sr_ref_LV0101', 'irrad_LV0102', 'temp_deg_LV0102', 'lvrt_ext_LV0102', 'ramp_enable_LV0102', 'p_s_ppc_LV0102', 'q_s_ppc_LV0102', 'i_sa_ref_LV0102', 'i_sr_ref_LV0102', 'irrad_LV0103', 'temp_deg_LV0103', 'lvrt_ext_LV0103', 'ramp_enable_LV0103', 'p_s_ppc_LV0103', 'q_s_ppc_LV0103', 'i_sa_ref_LV0103', 'i_sr_ref_LV0103', 'irrad_LV0104', 'temp_deg_LV0104', 'lvrt_ext_LV0104', 'ramp_enable_LV0104', 'p_s_ppc_LV0104', 'q_s_ppc_LV0104', 'i_sa_ref_LV0104', 'i_sr_ref_LV0104', 'irrad_LV0105', 'temp_deg_LV0105', 'lvrt_ext_LV0105', 'ramp_enable_LV0105', 'p_s_ppc_LV0105', 'q_s_ppc_LV0105', 'i_sa_ref_LV0105', 'i_sr_ref_LV0105', 'irrad_LV0106', 'temp_deg_LV0106', 'lvrt_ext_LV0106', 'ramp_enable_LV0106', 'p_s_ppc_LV0106', 'q_s_ppc_LV0106', 'i_sa_ref_LV0106', 'i_sr_ref_LV0106', 'irrad_LV0107', 'temp_deg_LV0107', 'lvrt_ext_LV0107', 'ramp_enable_LV0107', 'p_s_ppc_LV0107', 'q_s_ppc_LV0107', 'i_sa_ref_LV0107', 'i_sr_ref_LV0107', 'irrad_LV0108', 'temp_deg_LV0108', 'lvrt_ext_LV0108', 'ramp_enable_LV0108', 'p_s_ppc_LV0108', 'q_s_ppc_LV0108', 'i_sa_ref_LV0108', 'i_sr_ref_LV0108', 'irrad_LV0109', 'temp_deg_LV0109', 'lvrt_ext_LV0109', 'ramp_enable_LV0109', 'p_s_ppc_LV0109', 'q_s_ppc_LV0109', 'i_sa_ref_LV0109', 'i_sr_ref_LV0109', 'irrad_LV0110', 'temp_deg_LV0110', 'lvrt_ext_LV0110', 'ramp_enable_LV0110', 'p_s_ppc_LV0110', 'q_s_ppc_LV0110', 'i_sa_ref_LV0110', 'i_sr_ref_LV0110', 'irrad_LV0111', 'temp_deg_LV0111', 'lvrt_ext_LV0111', 'ramp_enable_LV0111', 'p_s_ppc_LV0111', 'q_s_ppc_LV0111', 'i_sa_ref_LV0111', 'i_sr_ref_LV0111', 'irrad_LV0112', 'temp_deg_LV0112', 'lvrt_ext_LV0112', 'ramp_enable_LV0112', 'p_s_ppc_LV0112', 'q_s_ppc_LV0112', 'i_sa_ref_LV0112', 'i_sr_ref_LV0112', 'irrad_LV0113', 'temp_deg_LV0113', 'lvrt_ext_LV0113', 'ramp_enable_LV0113', 'p_s_ppc_LV0113', 'q_s_ppc_LV0113', 'i_sa_ref_LV0113', 'i_sr_ref_LV0113', 'irrad_LV0114', 'temp_deg_LV0114', 'lvrt_ext_LV0114', 'ramp_enable_LV0114', 'p_s_ppc_LV0114', 'q_s_ppc_LV0114', 'i_sa_ref_LV0114', 'i_sr_ref_LV0114', 'irrad_LV0115', 'temp_deg_LV0115', 'lvrt_ext_LV0115', 'ramp_enable_LV0115', 'p_s_ppc_LV0115', 'q_s_ppc_LV0115', 'i_sa_ref_LV0115', 'i_sr_ref_LV0115', 'irrad_LV0116', 'temp_deg_LV0116', 'lvrt_ext_LV0116', 'ramp_enable_LV0116', 'p_s_ppc_LV0116', 'q_s_ppc_LV0116', 'i_sa_ref_LV0116', 'i_sr_ref_LV0116', 'irrad_LV0117', 'temp_deg_LV0117', 'lvrt_ext_LV0117', 'ramp_enable_LV0117', 'p_s_ppc_LV0117', 'q_s_ppc_LV0117', 'i_sa_ref_LV0117', 'i_sr_ref_LV0117', 'irrad_LV0118', 'temp_deg_LV0118', 'lvrt_ext_LV0118', 'ramp_enable_LV0118', 'p_s_ppc_LV0118', 'q_s_ppc_LV0118', 'i_sa_ref_LV0118', 'i_sr_ref_LV0118', 'irrad_LV0119', 'temp_deg_LV0119', 'lvrt_ext_LV0119', 'ramp_enable_LV0119', 'p_s_ppc_LV0119', 'q_s_ppc_LV0119', 'i_sa_ref_LV0119', 'i_sr_ref_LV0119', 'irrad_LV0120', 'temp_deg_LV0120', 'lvrt_ext_LV0120', 'ramp_enable_LV0120', 'p_s_ppc_LV0120', 'q_s_ppc_LV0120', 'i_sa_ref_LV0120', 'i_sr_ref_LV0120', 'irrad_LV0201', 'temp_deg_LV0201', 'lvrt_ext_LV0201', 'ramp_enable_LV0201', 'p_s_ppc_LV0201', 'q_s_ppc_LV0201', 'i_sa_ref_LV0201', 'i_sr_ref_LV0201', 'irrad_LV0202', 'temp_deg_LV0202', 'lvrt_ext_LV0202', 'ramp_enable_LV0202', 'p_s_ppc_LV0202', 'q_s_ppc_LV0202', 'i_sa_ref_LV0202', 'i_sr_ref_LV0202', 'irrad_LV0203', 'temp_deg_LV0203', 'lvrt_ext_LV0203', 'ramp_enable_LV0203', 'p_s_ppc_LV0203', 'q_s_ppc_LV0203', 'i_sa_ref_LV0203', 'i_sr_ref_LV0203', 'irrad_LV0204', 'temp_deg_LV0204', 'lvrt_ext_LV0204', 'ramp_enable_LV0204', 'p_s_ppc_LV0204', 'q_s_ppc_LV0204', 'i_sa_ref_LV0204', 'i_sr_ref_LV0204', 'irrad_LV0205', 'temp_deg_LV0205', 'lvrt_ext_LV0205', 'ramp_enable_LV0205', 'p_s_ppc_LV0205', 'q_s_ppc_LV0205', 'i_sa_ref_LV0205', 'i_sr_ref_LV0205', 'irrad_LV0206', 'temp_deg_LV0206', 'lvrt_ext_LV0206', 'ramp_enable_LV0206', 'p_s_ppc_LV0206', 'q_s_ppc_LV0206', 'i_sa_ref_LV0206', 'i_sr_ref_LV0206', 'irrad_LV0207', 'temp_deg_LV0207', 'lvrt_ext_LV0207', 'ramp_enable_LV0207', 'p_s_ppc_LV0207', 'q_s_ppc_LV0207', 'i_sa_ref_LV0207', 'i_sr_ref_LV0207', 'irrad_LV0208', 'temp_deg_LV0208', 'lvrt_ext_LV0208', 'ramp_enable_LV0208', 'p_s_ppc_LV0208', 'q_s_ppc_LV0208', 'i_sa_ref_LV0208', 'i_sr_ref_LV0208', 'irrad_LV0209', 'temp_deg_LV0209', 'lvrt_ext_LV0209', 'ramp_enable_LV0209', 'p_s_ppc_LV0209', 'q_s_ppc_LV0209', 'i_sa_ref_LV0209', 'i_sr_ref_LV0209', 'irrad_LV0210', 'temp_deg_LV0210', 'lvrt_ext_LV0210', 'ramp_enable_LV0210', 'p_s_ppc_LV0210', 'q_s_ppc_LV0210', 'i_sa_ref_LV0210', 'i_sr_ref_LV0210', 'irrad_LV0211', 'temp_deg_LV0211', 'lvrt_ext_LV0211', 'ramp_enable_LV0211', 'p_s_ppc_LV0211', 'q_s_ppc_LV0211', 'i_sa_ref_LV0211', 'i_sr_ref_LV0211', 'irrad_LV0212', 'temp_deg_LV0212', 'lvrt_ext_LV0212', 'ramp_enable_LV0212', 'p_s_ppc_LV0212', 'q_s_ppc_LV0212', 'i_sa_ref_LV0212', 'i_sr_ref_LV0212', 'irrad_LV0213', 'temp_deg_LV0213', 'lvrt_ext_LV0213', 'ramp_enable_LV0213', 'p_s_ppc_LV0213', 'q_s_ppc_LV0213', 'i_sa_ref_LV0213', 'i_sr_ref_LV0213', 'irrad_LV0214', 'temp_deg_LV0214', 'lvrt_ext_LV0214', 'ramp_enable_LV0214', 'p_s_ppc_LV0214', 'q_s_ppc_LV0214', 'i_sa_ref_LV0214', 'i_sr_ref_LV0214', 'irrad_LV0215', 'temp_deg_LV0215', 'lvrt_ext_LV0215', 'ramp_enable_LV0215', 'p_s_ppc_LV0215', 'q_s_ppc_LV0215', 'i_sa_ref_LV0215', 'i_sr_ref_LV0215', 'irrad_LV0216', 'temp_deg_LV0216', 'lvrt_ext_LV0216', 'ramp_enable_LV0216', 'p_s_ppc_LV0216', 'q_s_ppc_LV0216', 'i_sa_ref_LV0216', 'i_sr_ref_LV0216', 'irrad_LV0217', 'temp_deg_LV0217', 'lvrt_ext_LV0217', 'ramp_enable_LV0217', 'p_s_ppc_LV0217', 'q_s_ppc_LV0217', 'i_sa_ref_LV0217', 'i_sr_ref_LV0217', 'irrad_LV0218', 'temp_deg_LV0218', 'lvrt_ext_LV0218', 'ramp_enable_LV0218', 'p_s_ppc_LV0218', 'q_s_ppc_LV0218', 'i_sa_ref_LV0218', 'i_sr_ref_LV0218', 'irrad_LV0219', 'temp_deg_LV0219', 'lvrt_ext_LV0219', 'ramp_enable_LV0219', 'p_s_ppc_LV0219', 'q_s_ppc_LV0219', 'i_sa_ref_LV0219', 'i_sr_ref_LV0219', 'irrad_LV0220', 'temp_deg_LV0220', 'lvrt_ext_LV0220', 'ramp_enable_LV0220', 'p_s_ppc_LV0220', 'q_s_ppc_LV0220', 'i_sa_ref_LV0220', 'i_sr_ref_LV0220'] 
        self.inputs_ini_values_list  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0, 1.0, 1.0, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0] 
        self.inputs_run_list = ['P_POI_MV', 'Q_POI_MV', 'P_POI', 'Q_POI', 'P_GRID', 'Q_GRID', 'P_BESS', 'Q_BESS', 'P_LV0101', 'Q_LV0101', 'P_MV0101', 'Q_MV0101', 'P_LV0102', 'Q_LV0102', 'P_MV0102', 'Q_MV0102', 'P_LV0103', 'Q_LV0103', 'P_MV0103', 'Q_MV0103', 'P_LV0104', 'Q_LV0104', 'P_MV0104', 'Q_MV0104', 'P_LV0105', 'Q_LV0105', 'P_MV0105', 'Q_MV0105', 'P_LV0106', 'Q_LV0106', 'P_MV0106', 'Q_MV0106', 'P_LV0107', 'Q_LV0107', 'P_MV0107', 'Q_MV0107', 'P_LV0108', 'Q_LV0108', 'P_MV0108', 'Q_MV0108', 'P_LV0109', 'Q_LV0109', 'P_MV0109', 'Q_MV0109', 'P_LV0110', 'Q_LV0110', 'P_MV0110', 'Q_MV0110', 'P_LV0111', 'Q_LV0111', 'P_MV0111', 'Q_MV0111', 'P_LV0112', 'Q_LV0112', 'P_MV0112', 'Q_MV0112', 'P_LV0113', 'Q_LV0113', 'P_MV0113', 'Q_MV0113', 'P_LV0114', 'Q_LV0114', 'P_MV0114', 'Q_MV0114', 'P_LV0115', 'Q_LV0115', 'P_MV0115', 'Q_MV0115', 'P_LV0116', 'Q_LV0116', 'P_MV0116', 'Q_MV0116', 'P_LV0117', 'Q_LV0117', 'P_MV0117', 'Q_MV0117', 'P_LV0118', 'Q_LV0118', 'P_MV0118', 'Q_MV0118', 'P_LV0119', 'Q_LV0119', 'P_MV0119', 'Q_MV0119', 'P_LV0120', 'Q_LV0120', 'P_MV0120', 'Q_MV0120', 'P_LV0201', 'Q_LV0201', 'P_MV0201', 'Q_MV0201', 'P_LV0202', 'Q_LV0202', 'P_MV0202', 'Q_MV0202', 'P_LV0203', 'Q_LV0203', 'P_MV0203', 'Q_MV0203', 'P_LV0204', 'Q_LV0204', 'P_MV0204', 'Q_MV0204', 'P_LV0205', 'Q_LV0205', 'P_MV0205', 'Q_MV0205', 'P_LV0206', 'Q_LV0206', 'P_MV0206', 'Q_MV0206', 'P_LV0207', 'Q_LV0207', 'P_MV0207', 'Q_MV0207', 'P_LV0208', 'Q_LV0208', 'P_MV0208', 'Q_MV0208', 'P_LV0209', 'Q_LV0209', 'P_MV0209', 'Q_MV0209', 'P_LV0210', 'Q_LV0210', 'P_MV0210', 'Q_MV0210', 'P_LV0211', 'Q_LV0211', 'P_MV0211', 'Q_MV0211', 'P_LV0212', 'Q_LV0212', 'P_MV0212', 'Q_MV0212', 'P_LV0213', 'Q_LV0213', 'P_MV0213', 'Q_MV0213', 'P_LV0214', 'Q_LV0214', 'P_MV0214', 'Q_MV0214', 'P_LV0215', 'Q_LV0215', 'P_MV0215', 'Q_MV0215', 'P_LV0216', 'Q_LV0216', 'P_MV0216', 'Q_MV0216', 'P_LV0217', 'Q_LV0217', 'P_MV0217', 'Q_MV0217', 'P_LV0218', 'Q_LV0218', 'P_MV0218', 'Q_MV0218', 'P_LV0219', 'Q_LV0219', 'P_MV0219', 'Q_MV0219', 'P_LV0220', 'Q_LV0220', 'P_MV0220', 'Q_MV0220', 'p_s_ref_BESS', 'q_s_ref_BESS', 'soc_ref_BESS', 'alpha_GRID', 'v_ref_GRID', 'omega_ref_GRID', 'delta_ref_GRID', 'phi_GRID', 'rocov_GRID', 'irrad_LV0101', 'temp_deg_LV0101', 'lvrt_ext_LV0101', 'ramp_enable_LV0101', 'p_s_ppc_LV0101', 'q_s_ppc_LV0101', 'i_sa_ref_LV0101', 'i_sr_ref_LV0101', 'irrad_LV0102', 'temp_deg_LV0102', 'lvrt_ext_LV0102', 'ramp_enable_LV0102', 'p_s_ppc_LV0102', 'q_s_ppc_LV0102', 'i_sa_ref_LV0102', 'i_sr_ref_LV0102', 'irrad_LV0103', 'temp_deg_LV0103', 'lvrt_ext_LV0103', 'ramp_enable_LV0103', 'p_s_ppc_LV0103', 'q_s_ppc_LV0103', 'i_sa_ref_LV0103', 'i_sr_ref_LV0103', 'irrad_LV0104', 'temp_deg_LV0104', 'lvrt_ext_LV0104', 'ramp_enable_LV0104', 'p_s_ppc_LV0104', 'q_s_ppc_LV0104', 'i_sa_ref_LV0104', 'i_sr_ref_LV0104', 'irrad_LV0105', 'temp_deg_LV0105', 'lvrt_ext_LV0105', 'ramp_enable_LV0105', 'p_s_ppc_LV0105', 'q_s_ppc_LV0105', 'i_sa_ref_LV0105', 'i_sr_ref_LV0105', 'irrad_LV0106', 'temp_deg_LV0106', 'lvrt_ext_LV0106', 'ramp_enable_LV0106', 'p_s_ppc_LV0106', 'q_s_ppc_LV0106', 'i_sa_ref_LV0106', 'i_sr_ref_LV0106', 'irrad_LV0107', 'temp_deg_LV0107', 'lvrt_ext_LV0107', 'ramp_enable_LV0107', 'p_s_ppc_LV0107', 'q_s_ppc_LV0107', 'i_sa_ref_LV0107', 'i_sr_ref_LV0107', 'irrad_LV0108', 'temp_deg_LV0108', 'lvrt_ext_LV0108', 'ramp_enable_LV0108', 'p_s_ppc_LV0108', 'q_s_ppc_LV0108', 'i_sa_ref_LV0108', 'i_sr_ref_LV0108', 'irrad_LV0109', 'temp_deg_LV0109', 'lvrt_ext_LV0109', 'ramp_enable_LV0109', 'p_s_ppc_LV0109', 'q_s_ppc_LV0109', 'i_sa_ref_LV0109', 'i_sr_ref_LV0109', 'irrad_LV0110', 'temp_deg_LV0110', 'lvrt_ext_LV0110', 'ramp_enable_LV0110', 'p_s_ppc_LV0110', 'q_s_ppc_LV0110', 'i_sa_ref_LV0110', 'i_sr_ref_LV0110', 'irrad_LV0111', 'temp_deg_LV0111', 'lvrt_ext_LV0111', 'ramp_enable_LV0111', 'p_s_ppc_LV0111', 'q_s_ppc_LV0111', 'i_sa_ref_LV0111', 'i_sr_ref_LV0111', 'irrad_LV0112', 'temp_deg_LV0112', 'lvrt_ext_LV0112', 'ramp_enable_LV0112', 'p_s_ppc_LV0112', 'q_s_ppc_LV0112', 'i_sa_ref_LV0112', 'i_sr_ref_LV0112', 'irrad_LV0113', 'temp_deg_LV0113', 'lvrt_ext_LV0113', 'ramp_enable_LV0113', 'p_s_ppc_LV0113', 'q_s_ppc_LV0113', 'i_sa_ref_LV0113', 'i_sr_ref_LV0113', 'irrad_LV0114', 'temp_deg_LV0114', 'lvrt_ext_LV0114', 'ramp_enable_LV0114', 'p_s_ppc_LV0114', 'q_s_ppc_LV0114', 'i_sa_ref_LV0114', 'i_sr_ref_LV0114', 'irrad_LV0115', 'temp_deg_LV0115', 'lvrt_ext_LV0115', 'ramp_enable_LV0115', 'p_s_ppc_LV0115', 'q_s_ppc_LV0115', 'i_sa_ref_LV0115', 'i_sr_ref_LV0115', 'irrad_LV0116', 'temp_deg_LV0116', 'lvrt_ext_LV0116', 'ramp_enable_LV0116', 'p_s_ppc_LV0116', 'q_s_ppc_LV0116', 'i_sa_ref_LV0116', 'i_sr_ref_LV0116', 'irrad_LV0117', 'temp_deg_LV0117', 'lvrt_ext_LV0117', 'ramp_enable_LV0117', 'p_s_ppc_LV0117', 'q_s_ppc_LV0117', 'i_sa_ref_LV0117', 'i_sr_ref_LV0117', 'irrad_LV0118', 'temp_deg_LV0118', 'lvrt_ext_LV0118', 'ramp_enable_LV0118', 'p_s_ppc_LV0118', 'q_s_ppc_LV0118', 'i_sa_ref_LV0118', 'i_sr_ref_LV0118', 'irrad_LV0119', 'temp_deg_LV0119', 'lvrt_ext_LV0119', 'ramp_enable_LV0119', 'p_s_ppc_LV0119', 'q_s_ppc_LV0119', 'i_sa_ref_LV0119', 'i_sr_ref_LV0119', 'irrad_LV0120', 'temp_deg_LV0120', 'lvrt_ext_LV0120', 'ramp_enable_LV0120', 'p_s_ppc_LV0120', 'q_s_ppc_LV0120', 'i_sa_ref_LV0120', 'i_sr_ref_LV0120', 'irrad_LV0201', 'temp_deg_LV0201', 'lvrt_ext_LV0201', 'ramp_enable_LV0201', 'p_s_ppc_LV0201', 'q_s_ppc_LV0201', 'i_sa_ref_LV0201', 'i_sr_ref_LV0201', 'irrad_LV0202', 'temp_deg_LV0202', 'lvrt_ext_LV0202', 'ramp_enable_LV0202', 'p_s_ppc_LV0202', 'q_s_ppc_LV0202', 'i_sa_ref_LV0202', 'i_sr_ref_LV0202', 'irrad_LV0203', 'temp_deg_LV0203', 'lvrt_ext_LV0203', 'ramp_enable_LV0203', 'p_s_ppc_LV0203', 'q_s_ppc_LV0203', 'i_sa_ref_LV0203', 'i_sr_ref_LV0203', 'irrad_LV0204', 'temp_deg_LV0204', 'lvrt_ext_LV0204', 'ramp_enable_LV0204', 'p_s_ppc_LV0204', 'q_s_ppc_LV0204', 'i_sa_ref_LV0204', 'i_sr_ref_LV0204', 'irrad_LV0205', 'temp_deg_LV0205', 'lvrt_ext_LV0205', 'ramp_enable_LV0205', 'p_s_ppc_LV0205', 'q_s_ppc_LV0205', 'i_sa_ref_LV0205', 'i_sr_ref_LV0205', 'irrad_LV0206', 'temp_deg_LV0206', 'lvrt_ext_LV0206', 'ramp_enable_LV0206', 'p_s_ppc_LV0206', 'q_s_ppc_LV0206', 'i_sa_ref_LV0206', 'i_sr_ref_LV0206', 'irrad_LV0207', 'temp_deg_LV0207', 'lvrt_ext_LV0207', 'ramp_enable_LV0207', 'p_s_ppc_LV0207', 'q_s_ppc_LV0207', 'i_sa_ref_LV0207', 'i_sr_ref_LV0207', 'irrad_LV0208', 'temp_deg_LV0208', 'lvrt_ext_LV0208', 'ramp_enable_LV0208', 'p_s_ppc_LV0208', 'q_s_ppc_LV0208', 'i_sa_ref_LV0208', 'i_sr_ref_LV0208', 'irrad_LV0209', 'temp_deg_LV0209', 'lvrt_ext_LV0209', 'ramp_enable_LV0209', 'p_s_ppc_LV0209', 'q_s_ppc_LV0209', 'i_sa_ref_LV0209', 'i_sr_ref_LV0209', 'irrad_LV0210', 'temp_deg_LV0210', 'lvrt_ext_LV0210', 'ramp_enable_LV0210', 'p_s_ppc_LV0210', 'q_s_ppc_LV0210', 'i_sa_ref_LV0210', 'i_sr_ref_LV0210', 'irrad_LV0211', 'temp_deg_LV0211', 'lvrt_ext_LV0211', 'ramp_enable_LV0211', 'p_s_ppc_LV0211', 'q_s_ppc_LV0211', 'i_sa_ref_LV0211', 'i_sr_ref_LV0211', 'irrad_LV0212', 'temp_deg_LV0212', 'lvrt_ext_LV0212', 'ramp_enable_LV0212', 'p_s_ppc_LV0212', 'q_s_ppc_LV0212', 'i_sa_ref_LV0212', 'i_sr_ref_LV0212', 'irrad_LV0213', 'temp_deg_LV0213', 'lvrt_ext_LV0213', 'ramp_enable_LV0213', 'p_s_ppc_LV0213', 'q_s_ppc_LV0213', 'i_sa_ref_LV0213', 'i_sr_ref_LV0213', 'irrad_LV0214', 'temp_deg_LV0214', 'lvrt_ext_LV0214', 'ramp_enable_LV0214', 'p_s_ppc_LV0214', 'q_s_ppc_LV0214', 'i_sa_ref_LV0214', 'i_sr_ref_LV0214', 'irrad_LV0215', 'temp_deg_LV0215', 'lvrt_ext_LV0215', 'ramp_enable_LV0215', 'p_s_ppc_LV0215', 'q_s_ppc_LV0215', 'i_sa_ref_LV0215', 'i_sr_ref_LV0215', 'irrad_LV0216', 'temp_deg_LV0216', 'lvrt_ext_LV0216', 'ramp_enable_LV0216', 'p_s_ppc_LV0216', 'q_s_ppc_LV0216', 'i_sa_ref_LV0216', 'i_sr_ref_LV0216', 'irrad_LV0217', 'temp_deg_LV0217', 'lvrt_ext_LV0217', 'ramp_enable_LV0217', 'p_s_ppc_LV0217', 'q_s_ppc_LV0217', 'i_sa_ref_LV0217', 'i_sr_ref_LV0217', 'irrad_LV0218', 'temp_deg_LV0218', 'lvrt_ext_LV0218', 'ramp_enable_LV0218', 'p_s_ppc_LV0218', 'q_s_ppc_LV0218', 'i_sa_ref_LV0218', 'i_sr_ref_LV0218', 'irrad_LV0219', 'temp_deg_LV0219', 'lvrt_ext_LV0219', 'ramp_enable_LV0219', 'p_s_ppc_LV0219', 'q_s_ppc_LV0219', 'i_sa_ref_LV0219', 'i_sr_ref_LV0219', 'irrad_LV0220', 'temp_deg_LV0220', 'lvrt_ext_LV0220', 'ramp_enable_LV0220', 'p_s_ppc_LV0220', 'q_s_ppc_LV0220', 'i_sa_ref_LV0220', 'i_sr_ref_LV0220'] 
        self.inputs_run_values_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0, 1.0, 1.0, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0] 
        self.outputs_list = ['V_POI_MV', 'V_POI', 'V_GRID', 'V_BESS', 'V_LV0101', 'V_MV0101', 'V_LV0102', 'V_MV0102', 'V_LV0103', 'V_MV0103', 'V_LV0104', 'V_MV0104', 'V_LV0105', 'V_MV0105', 'V_LV0106', 'V_MV0106', 'V_LV0107', 'V_MV0107', 'V_LV0108', 'V_MV0108', 'V_LV0109', 'V_MV0109', 'V_LV0110', 'V_MV0110', 'V_LV0111', 'V_MV0111', 'V_LV0112', 'V_MV0112', 'V_LV0113', 'V_MV0113', 'V_LV0114', 'V_MV0114', 'V_LV0115', 'V_MV0115', 'V_LV0116', 'V_MV0116', 'V_LV0117', 'V_MV0117', 'V_LV0118', 'V_MV0118', 'V_LV0119', 'V_MV0119', 'V_LV0120', 'V_MV0120', 'V_LV0201', 'V_MV0201', 'V_LV0202', 'V_MV0202', 'V_LV0203', 'V_MV0203', 'V_LV0204', 'V_MV0204', 'V_LV0205', 'V_MV0205', 'V_LV0206', 'V_MV0206', 'V_LV0207', 'V_MV0207', 'V_LV0208', 'V_MV0208', 'V_LV0209', 'V_MV0209', 'V_LV0210', 'V_MV0210', 'V_LV0211', 'V_MV0211', 'V_LV0212', 'V_MV0212', 'V_LV0213', 'V_MV0213', 'V_LV0214', 'V_MV0214', 'V_LV0215', 'V_MV0215', 'V_LV0216', 'V_MV0216', 'V_LV0217', 'V_MV0217', 'V_LV0218', 'V_MV0218', 'V_LV0219', 'V_MV0219', 'V_LV0220', 'V_MV0220', 'p_line_POI_GRID', 'q_line_POI_GRID', 'p_line_GRID_POI', 'q_line_GRID_POI', 'p_line_BESS_POI_MV', 'q_line_BESS_POI_MV', 'p_line_POI_MV_BESS', 'q_line_POI_MV_BESS', 'p_line_MV0101_POI_MV', 'q_line_MV0101_POI_MV', 'p_line_POI_MV_MV0101', 'q_line_POI_MV_MV0101', 'p_line_MV0201_POI_MV', 'q_line_MV0201_POI_MV', 'p_line_POI_MV_MV0201', 'q_line_POI_MV_MV0201', 'p_loss_BESS', 'i_s_BESS', 'e_BESS', 'i_dc_BESS', 'p_s_BESS', 'q_s_BESS', 'alpha_GRID', 'Dv_GRID', 'm_ref_LV0101', 'v_sd_LV0101', 'v_sq_LV0101', 'lvrt_LV0101', 'm_ref_LV0102', 'v_sd_LV0102', 'v_sq_LV0102', 'lvrt_LV0102', 'm_ref_LV0103', 'v_sd_LV0103', 'v_sq_LV0103', 'lvrt_LV0103', 'm_ref_LV0104', 'v_sd_LV0104', 'v_sq_LV0104', 'lvrt_LV0104', 'm_ref_LV0105', 'v_sd_LV0105', 'v_sq_LV0105', 'lvrt_LV0105', 'm_ref_LV0106', 'v_sd_LV0106', 'v_sq_LV0106', 'lvrt_LV0106', 'm_ref_LV0107', 'v_sd_LV0107', 'v_sq_LV0107', 'lvrt_LV0107', 'm_ref_LV0108', 'v_sd_LV0108', 'v_sq_LV0108', 'lvrt_LV0108', 'm_ref_LV0109', 'v_sd_LV0109', 'v_sq_LV0109', 'lvrt_LV0109', 'm_ref_LV0110', 'v_sd_LV0110', 'v_sq_LV0110', 'lvrt_LV0110', 'm_ref_LV0111', 'v_sd_LV0111', 'v_sq_LV0111', 'lvrt_LV0111', 'm_ref_LV0112', 'v_sd_LV0112', 'v_sq_LV0112', 'lvrt_LV0112', 'm_ref_LV0113', 'v_sd_LV0113', 'v_sq_LV0113', 'lvrt_LV0113', 'm_ref_LV0114', 'v_sd_LV0114', 'v_sq_LV0114', 'lvrt_LV0114', 'm_ref_LV0115', 'v_sd_LV0115', 'v_sq_LV0115', 'lvrt_LV0115', 'm_ref_LV0116', 'v_sd_LV0116', 'v_sq_LV0116', 'lvrt_LV0116', 'm_ref_LV0117', 'v_sd_LV0117', 'v_sq_LV0117', 'lvrt_LV0117', 'm_ref_LV0118', 'v_sd_LV0118', 'v_sq_LV0118', 'lvrt_LV0118', 'm_ref_LV0119', 'v_sd_LV0119', 'v_sq_LV0119', 'lvrt_LV0119', 'm_ref_LV0120', 'v_sd_LV0120', 'v_sq_LV0120', 'lvrt_LV0120', 'm_ref_LV0201', 'v_sd_LV0201', 'v_sq_LV0201', 'lvrt_LV0201', 'm_ref_LV0202', 'v_sd_LV0202', 'v_sq_LV0202', 'lvrt_LV0202', 'm_ref_LV0203', 'v_sd_LV0203', 'v_sq_LV0203', 'lvrt_LV0203', 'm_ref_LV0204', 'v_sd_LV0204', 'v_sq_LV0204', 'lvrt_LV0204', 'm_ref_LV0205', 'v_sd_LV0205', 'v_sq_LV0205', 'lvrt_LV0205', 'm_ref_LV0206', 'v_sd_LV0206', 'v_sq_LV0206', 'lvrt_LV0206', 'm_ref_LV0207', 'v_sd_LV0207', 'v_sq_LV0207', 'lvrt_LV0207', 'm_ref_LV0208', 'v_sd_LV0208', 'v_sq_LV0208', 'lvrt_LV0208', 'm_ref_LV0209', 'v_sd_LV0209', 'v_sq_LV0209', 'lvrt_LV0209', 'm_ref_LV0210', 'v_sd_LV0210', 'v_sq_LV0210', 'lvrt_LV0210', 'm_ref_LV0211', 'v_sd_LV0211', 'v_sq_LV0211', 'lvrt_LV0211', 'm_ref_LV0212', 'v_sd_LV0212', 'v_sq_LV0212', 'lvrt_LV0212', 'm_ref_LV0213', 'v_sd_LV0213', 'v_sq_LV0213', 'lvrt_LV0213', 'm_ref_LV0214', 'v_sd_LV0214', 'v_sq_LV0214', 'lvrt_LV0214', 'm_ref_LV0215', 'v_sd_LV0215', 'v_sq_LV0215', 'lvrt_LV0215', 'm_ref_LV0216', 'v_sd_LV0216', 'v_sq_LV0216', 'lvrt_LV0216', 'm_ref_LV0217', 'v_sd_LV0217', 'v_sq_LV0217', 'lvrt_LV0217', 'm_ref_LV0218', 'v_sd_LV0218', 'v_sq_LV0218', 'lvrt_LV0218', 'm_ref_LV0219', 'v_sd_LV0219', 'v_sq_LV0219', 'lvrt_LV0219', 'm_ref_LV0220', 'v_sd_LV0220', 'v_sq_LV0220', 'lvrt_LV0220'] 
        self.x_list = ['soc_BESS', 'xi_soc_BESS', 'delta_GRID', 'Domega_GRID', 'Dv_GRID', 'x_p_lp1_LV0101', 'x_p_lp2_LV0101', 'x_q_lp1_LV0101', 'x_q_lp2_LV0101', 'x_p_lp1_LV0102', 'x_p_lp2_LV0102', 'x_q_lp1_LV0102', 'x_q_lp2_LV0102', 'x_p_lp1_LV0103', 'x_p_lp2_LV0103', 'x_q_lp1_LV0103', 'x_q_lp2_LV0103', 'x_p_lp1_LV0104', 'x_p_lp2_LV0104', 'x_q_lp1_LV0104', 'x_q_lp2_LV0104', 'x_p_lp1_LV0105', 'x_p_lp2_LV0105', 'x_q_lp1_LV0105', 'x_q_lp2_LV0105', 'x_p_lp1_LV0106', 'x_p_lp2_LV0106', 'x_q_lp1_LV0106', 'x_q_lp2_LV0106', 'x_p_lp1_LV0107', 'x_p_lp2_LV0107', 'x_q_lp1_LV0107', 'x_q_lp2_LV0107', 'x_p_lp1_LV0108', 'x_p_lp2_LV0108', 'x_q_lp1_LV0108', 'x_q_lp2_LV0108', 'x_p_lp1_LV0109', 'x_p_lp2_LV0109', 'x_q_lp1_LV0109', 'x_q_lp2_LV0109', 'x_p_lp1_LV0110', 'x_p_lp2_LV0110', 'x_q_lp1_LV0110', 'x_q_lp2_LV0110', 'x_p_lp1_LV0111', 'x_p_lp2_LV0111', 'x_q_lp1_LV0111', 'x_q_lp2_LV0111', 'x_p_lp1_LV0112', 'x_p_lp2_LV0112', 'x_q_lp1_LV0112', 'x_q_lp2_LV0112', 'x_p_lp1_LV0113', 'x_p_lp2_LV0113', 'x_q_lp1_LV0113', 'x_q_lp2_LV0113', 'x_p_lp1_LV0114', 'x_p_lp2_LV0114', 'x_q_lp1_LV0114', 'x_q_lp2_LV0114', 'x_p_lp1_LV0115', 'x_p_lp2_LV0115', 'x_q_lp1_LV0115', 'x_q_lp2_LV0115', 'x_p_lp1_LV0116', 'x_p_lp2_LV0116', 'x_q_lp1_LV0116', 'x_q_lp2_LV0116', 'x_p_lp1_LV0117', 'x_p_lp2_LV0117', 'x_q_lp1_LV0117', 'x_q_lp2_LV0117', 'x_p_lp1_LV0118', 'x_p_lp2_LV0118', 'x_q_lp1_LV0118', 'x_q_lp2_LV0118', 'x_p_lp1_LV0119', 'x_p_lp2_LV0119', 'x_q_lp1_LV0119', 'x_q_lp2_LV0119', 'x_p_lp1_LV0120', 'x_p_lp2_LV0120', 'x_q_lp1_LV0120', 'x_q_lp2_LV0120', 'x_p_lp1_LV0201', 'x_p_lp2_LV0201', 'x_q_lp1_LV0201', 'x_q_lp2_LV0201', 'x_p_lp1_LV0202', 'x_p_lp2_LV0202', 'x_q_lp1_LV0202', 'x_q_lp2_LV0202', 'x_p_lp1_LV0203', 'x_p_lp2_LV0203', 'x_q_lp1_LV0203', 'x_q_lp2_LV0203', 'x_p_lp1_LV0204', 'x_p_lp2_LV0204', 'x_q_lp1_LV0204', 'x_q_lp2_LV0204', 'x_p_lp1_LV0205', 'x_p_lp2_LV0205', 'x_q_lp1_LV0205', 'x_q_lp2_LV0205', 'x_p_lp1_LV0206', 'x_p_lp2_LV0206', 'x_q_lp1_LV0206', 'x_q_lp2_LV0206', 'x_p_lp1_LV0207', 'x_p_lp2_LV0207', 'x_q_lp1_LV0207', 'x_q_lp2_LV0207', 'x_p_lp1_LV0208', 'x_p_lp2_LV0208', 'x_q_lp1_LV0208', 'x_q_lp2_LV0208', 'x_p_lp1_LV0209', 'x_p_lp2_LV0209', 'x_q_lp1_LV0209', 'x_q_lp2_LV0209', 'x_p_lp1_LV0210', 'x_p_lp2_LV0210', 'x_q_lp1_LV0210', 'x_q_lp2_LV0210', 'x_p_lp1_LV0211', 'x_p_lp2_LV0211', 'x_q_lp1_LV0211', 'x_q_lp2_LV0211', 'x_p_lp1_LV0212', 'x_p_lp2_LV0212', 'x_q_lp1_LV0212', 'x_q_lp2_LV0212', 'x_p_lp1_LV0213', 'x_p_lp2_LV0213', 'x_q_lp1_LV0213', 'x_q_lp2_LV0213', 'x_p_lp1_LV0214', 'x_p_lp2_LV0214', 'x_q_lp1_LV0214', 'x_q_lp2_LV0214', 'x_p_lp1_LV0215', 'x_p_lp2_LV0215', 'x_q_lp1_LV0215', 'x_q_lp2_LV0215', 'x_p_lp1_LV0216', 'x_p_lp2_LV0216', 'x_q_lp1_LV0216', 'x_q_lp2_LV0216', 'x_p_lp1_LV0217', 'x_p_lp2_LV0217', 'x_q_lp1_LV0217', 'x_q_lp2_LV0217', 'x_p_lp1_LV0218', 'x_p_lp2_LV0218', 'x_q_lp1_LV0218', 'x_q_lp2_LV0218', 'x_p_lp1_LV0219', 'x_p_lp2_LV0219', 'x_q_lp1_LV0219', 'x_q_lp2_LV0219', 'x_p_lp1_LV0220', 'x_p_lp2_LV0220', 'x_q_lp1_LV0220', 'x_q_lp2_LV0220', 'xi_freq'] 
        self.y_run_list = ['V_POI_MV', 'theta_POI_MV', 'V_POI', 'theta_POI', 'V_GRID', 'theta_GRID', 'V_BESS', 'theta_BESS', 'V_LV0101', 'theta_LV0101', 'V_MV0101', 'theta_MV0101', 'V_LV0102', 'theta_LV0102', 'V_MV0102', 'theta_MV0102', 'V_LV0103', 'theta_LV0103', 'V_MV0103', 'theta_MV0103', 'V_LV0104', 'theta_LV0104', 'V_MV0104', 'theta_MV0104', 'V_LV0105', 'theta_LV0105', 'V_MV0105', 'theta_MV0105', 'V_LV0106', 'theta_LV0106', 'V_MV0106', 'theta_MV0106', 'V_LV0107', 'theta_LV0107', 'V_MV0107', 'theta_MV0107', 'V_LV0108', 'theta_LV0108', 'V_MV0108', 'theta_MV0108', 'V_LV0109', 'theta_LV0109', 'V_MV0109', 'theta_MV0109', 'V_LV0110', 'theta_LV0110', 'V_MV0110', 'theta_MV0110', 'V_LV0111', 'theta_LV0111', 'V_MV0111', 'theta_MV0111', 'V_LV0112', 'theta_LV0112', 'V_MV0112', 'theta_MV0112', 'V_LV0113', 'theta_LV0113', 'V_MV0113', 'theta_MV0113', 'V_LV0114', 'theta_LV0114', 'V_MV0114', 'theta_MV0114', 'V_LV0115', 'theta_LV0115', 'V_MV0115', 'theta_MV0115', 'V_LV0116', 'theta_LV0116', 'V_MV0116', 'theta_MV0116', 'V_LV0117', 'theta_LV0117', 'V_MV0117', 'theta_MV0117', 'V_LV0118', 'theta_LV0118', 'V_MV0118', 'theta_MV0118', 'V_LV0119', 'theta_LV0119', 'V_MV0119', 'theta_MV0119', 'V_LV0120', 'theta_LV0120', 'V_MV0120', 'theta_MV0120', 'V_LV0201', 'theta_LV0201', 'V_MV0201', 'theta_MV0201', 'V_LV0202', 'theta_LV0202', 'V_MV0202', 'theta_MV0202', 'V_LV0203', 'theta_LV0203', 'V_MV0203', 'theta_MV0203', 'V_LV0204', 'theta_LV0204', 'V_MV0204', 'theta_MV0204', 'V_LV0205', 'theta_LV0205', 'V_MV0205', 'theta_MV0205', 'V_LV0206', 'theta_LV0206', 'V_MV0206', 'theta_MV0206', 'V_LV0207', 'theta_LV0207', 'V_MV0207', 'theta_MV0207', 'V_LV0208', 'theta_LV0208', 'V_MV0208', 'theta_MV0208', 'V_LV0209', 'theta_LV0209', 'V_MV0209', 'theta_MV0209', 'V_LV0210', 'theta_LV0210', 'V_MV0210', 'theta_MV0210', 'V_LV0211', 'theta_LV0211', 'V_MV0211', 'theta_MV0211', 'V_LV0212', 'theta_LV0212', 'V_MV0212', 'theta_MV0212', 'V_LV0213', 'theta_LV0213', 'V_MV0213', 'theta_MV0213', 'V_LV0214', 'theta_LV0214', 'V_MV0214', 'theta_MV0214', 'V_LV0215', 'theta_LV0215', 'V_MV0215', 'theta_MV0215', 'V_LV0216', 'theta_LV0216', 'V_MV0216', 'theta_MV0216', 'V_LV0217', 'theta_LV0217', 'V_MV0217', 'theta_MV0217', 'V_LV0218', 'theta_LV0218', 'V_MV0218', 'theta_MV0218', 'V_LV0219', 'theta_LV0219', 'V_MV0219', 'theta_MV0219', 'V_LV0220', 'theta_LV0220', 'V_MV0220', 'theta_MV0220', 'p_dc_BESS', 'i_dc_BESS', 'v_dc_BESS', 'omega_GRID', 'i_d_GRID', 'i_q_GRID', 'p_s_GRID', 'q_s_GRID', 'v_dc_LV0101', 'i_sq_ref_LV0101', 'i_sd_ref_LV0101', 'i_sr_LV0101', 'i_si_LV0101', 'p_s_LV0101', 'q_s_LV0101', 'v_dc_LV0102', 'i_sq_ref_LV0102', 'i_sd_ref_LV0102', 'i_sr_LV0102', 'i_si_LV0102', 'p_s_LV0102', 'q_s_LV0102', 'v_dc_LV0103', 'i_sq_ref_LV0103', 'i_sd_ref_LV0103', 'i_sr_LV0103', 'i_si_LV0103', 'p_s_LV0103', 'q_s_LV0103', 'v_dc_LV0104', 'i_sq_ref_LV0104', 'i_sd_ref_LV0104', 'i_sr_LV0104', 'i_si_LV0104', 'p_s_LV0104', 'q_s_LV0104', 'v_dc_LV0105', 'i_sq_ref_LV0105', 'i_sd_ref_LV0105', 'i_sr_LV0105', 'i_si_LV0105', 'p_s_LV0105', 'q_s_LV0105', 'v_dc_LV0106', 'i_sq_ref_LV0106', 'i_sd_ref_LV0106', 'i_sr_LV0106', 'i_si_LV0106', 'p_s_LV0106', 'q_s_LV0106', 'v_dc_LV0107', 'i_sq_ref_LV0107', 'i_sd_ref_LV0107', 'i_sr_LV0107', 'i_si_LV0107', 'p_s_LV0107', 'q_s_LV0107', 'v_dc_LV0108', 'i_sq_ref_LV0108', 'i_sd_ref_LV0108', 'i_sr_LV0108', 'i_si_LV0108', 'p_s_LV0108', 'q_s_LV0108', 'v_dc_LV0109', 'i_sq_ref_LV0109', 'i_sd_ref_LV0109', 'i_sr_LV0109', 'i_si_LV0109', 'p_s_LV0109', 'q_s_LV0109', 'v_dc_LV0110', 'i_sq_ref_LV0110', 'i_sd_ref_LV0110', 'i_sr_LV0110', 'i_si_LV0110', 'p_s_LV0110', 'q_s_LV0110', 'v_dc_LV0111', 'i_sq_ref_LV0111', 'i_sd_ref_LV0111', 'i_sr_LV0111', 'i_si_LV0111', 'p_s_LV0111', 'q_s_LV0111', 'v_dc_LV0112', 'i_sq_ref_LV0112', 'i_sd_ref_LV0112', 'i_sr_LV0112', 'i_si_LV0112', 'p_s_LV0112', 'q_s_LV0112', 'v_dc_LV0113', 'i_sq_ref_LV0113', 'i_sd_ref_LV0113', 'i_sr_LV0113', 'i_si_LV0113', 'p_s_LV0113', 'q_s_LV0113', 'v_dc_LV0114', 'i_sq_ref_LV0114', 'i_sd_ref_LV0114', 'i_sr_LV0114', 'i_si_LV0114', 'p_s_LV0114', 'q_s_LV0114', 'v_dc_LV0115', 'i_sq_ref_LV0115', 'i_sd_ref_LV0115', 'i_sr_LV0115', 'i_si_LV0115', 'p_s_LV0115', 'q_s_LV0115', 'v_dc_LV0116', 'i_sq_ref_LV0116', 'i_sd_ref_LV0116', 'i_sr_LV0116', 'i_si_LV0116', 'p_s_LV0116', 'q_s_LV0116', 'v_dc_LV0117', 'i_sq_ref_LV0117', 'i_sd_ref_LV0117', 'i_sr_LV0117', 'i_si_LV0117', 'p_s_LV0117', 'q_s_LV0117', 'v_dc_LV0118', 'i_sq_ref_LV0118', 'i_sd_ref_LV0118', 'i_sr_LV0118', 'i_si_LV0118', 'p_s_LV0118', 'q_s_LV0118', 'v_dc_LV0119', 'i_sq_ref_LV0119', 'i_sd_ref_LV0119', 'i_sr_LV0119', 'i_si_LV0119', 'p_s_LV0119', 'q_s_LV0119', 'v_dc_LV0120', 'i_sq_ref_LV0120', 'i_sd_ref_LV0120', 'i_sr_LV0120', 'i_si_LV0120', 'p_s_LV0120', 'q_s_LV0120', 'v_dc_LV0201', 'i_sq_ref_LV0201', 'i_sd_ref_LV0201', 'i_sr_LV0201', 'i_si_LV0201', 'p_s_LV0201', 'q_s_LV0201', 'v_dc_LV0202', 'i_sq_ref_LV0202', 'i_sd_ref_LV0202', 'i_sr_LV0202', 'i_si_LV0202', 'p_s_LV0202', 'q_s_LV0202', 'v_dc_LV0203', 'i_sq_ref_LV0203', 'i_sd_ref_LV0203', 'i_sr_LV0203', 'i_si_LV0203', 'p_s_LV0203', 'q_s_LV0203', 'v_dc_LV0204', 'i_sq_ref_LV0204', 'i_sd_ref_LV0204', 'i_sr_LV0204', 'i_si_LV0204', 'p_s_LV0204', 'q_s_LV0204', 'v_dc_LV0205', 'i_sq_ref_LV0205', 'i_sd_ref_LV0205', 'i_sr_LV0205', 'i_si_LV0205', 'p_s_LV0205', 'q_s_LV0205', 'v_dc_LV0206', 'i_sq_ref_LV0206', 'i_sd_ref_LV0206', 'i_sr_LV0206', 'i_si_LV0206', 'p_s_LV0206', 'q_s_LV0206', 'v_dc_LV0207', 'i_sq_ref_LV0207', 'i_sd_ref_LV0207', 'i_sr_LV0207', 'i_si_LV0207', 'p_s_LV0207', 'q_s_LV0207', 'v_dc_LV0208', 'i_sq_ref_LV0208', 'i_sd_ref_LV0208', 'i_sr_LV0208', 'i_si_LV0208', 'p_s_LV0208', 'q_s_LV0208', 'v_dc_LV0209', 'i_sq_ref_LV0209', 'i_sd_ref_LV0209', 'i_sr_LV0209', 'i_si_LV0209', 'p_s_LV0209', 'q_s_LV0209', 'v_dc_LV0210', 'i_sq_ref_LV0210', 'i_sd_ref_LV0210', 'i_sr_LV0210', 'i_si_LV0210', 'p_s_LV0210', 'q_s_LV0210', 'v_dc_LV0211', 'i_sq_ref_LV0211', 'i_sd_ref_LV0211', 'i_sr_LV0211', 'i_si_LV0211', 'p_s_LV0211', 'q_s_LV0211', 'v_dc_LV0212', 'i_sq_ref_LV0212', 'i_sd_ref_LV0212', 'i_sr_LV0212', 'i_si_LV0212', 'p_s_LV0212', 'q_s_LV0212', 'v_dc_LV0213', 'i_sq_ref_LV0213', 'i_sd_ref_LV0213', 'i_sr_LV0213', 'i_si_LV0213', 'p_s_LV0213', 'q_s_LV0213', 'v_dc_LV0214', 'i_sq_ref_LV0214', 'i_sd_ref_LV0214', 'i_sr_LV0214', 'i_si_LV0214', 'p_s_LV0214', 'q_s_LV0214', 'v_dc_LV0215', 'i_sq_ref_LV0215', 'i_sd_ref_LV0215', 'i_sr_LV0215', 'i_si_LV0215', 'p_s_LV0215', 'q_s_LV0215', 'v_dc_LV0216', 'i_sq_ref_LV0216', 'i_sd_ref_LV0216', 'i_sr_LV0216', 'i_si_LV0216', 'p_s_LV0216', 'q_s_LV0216', 'v_dc_LV0217', 'i_sq_ref_LV0217', 'i_sd_ref_LV0217', 'i_sr_LV0217', 'i_si_LV0217', 'p_s_LV0217', 'q_s_LV0217', 'v_dc_LV0218', 'i_sq_ref_LV0218', 'i_sd_ref_LV0218', 'i_sr_LV0218', 'i_si_LV0218', 'p_s_LV0218', 'q_s_LV0218', 'v_dc_LV0219', 'i_sq_ref_LV0219', 'i_sd_ref_LV0219', 'i_sr_LV0219', 'i_si_LV0219', 'p_s_LV0219', 'q_s_LV0219', 'v_dc_LV0220', 'i_sq_ref_LV0220', 'i_sd_ref_LV0220', 'i_sr_LV0220', 'i_si_LV0220', 'p_s_LV0220', 'q_s_LV0220', 'omega_coi', 'p_agc'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['V_POI_MV', 'theta_POI_MV', 'V_POI', 'theta_POI', 'V_GRID', 'theta_GRID', 'V_BESS', 'theta_BESS', 'V_LV0101', 'theta_LV0101', 'V_MV0101', 'theta_MV0101', 'V_LV0102', 'theta_LV0102', 'V_MV0102', 'theta_MV0102', 'V_LV0103', 'theta_LV0103', 'V_MV0103', 'theta_MV0103', 'V_LV0104', 'theta_LV0104', 'V_MV0104', 'theta_MV0104', 'V_LV0105', 'theta_LV0105', 'V_MV0105', 'theta_MV0105', 'V_LV0106', 'theta_LV0106', 'V_MV0106', 'theta_MV0106', 'V_LV0107', 'theta_LV0107', 'V_MV0107', 'theta_MV0107', 'V_LV0108', 'theta_LV0108', 'V_MV0108', 'theta_MV0108', 'V_LV0109', 'theta_LV0109', 'V_MV0109', 'theta_MV0109', 'V_LV0110', 'theta_LV0110', 'V_MV0110', 'theta_MV0110', 'V_LV0111', 'theta_LV0111', 'V_MV0111', 'theta_MV0111', 'V_LV0112', 'theta_LV0112', 'V_MV0112', 'theta_MV0112', 'V_LV0113', 'theta_LV0113', 'V_MV0113', 'theta_MV0113', 'V_LV0114', 'theta_LV0114', 'V_MV0114', 'theta_MV0114', 'V_LV0115', 'theta_LV0115', 'V_MV0115', 'theta_MV0115', 'V_LV0116', 'theta_LV0116', 'V_MV0116', 'theta_MV0116', 'V_LV0117', 'theta_LV0117', 'V_MV0117', 'theta_MV0117', 'V_LV0118', 'theta_LV0118', 'V_MV0118', 'theta_MV0118', 'V_LV0119', 'theta_LV0119', 'V_MV0119', 'theta_MV0119', 'V_LV0120', 'theta_LV0120', 'V_MV0120', 'theta_MV0120', 'V_LV0201', 'theta_LV0201', 'V_MV0201', 'theta_MV0201', 'V_LV0202', 'theta_LV0202', 'V_MV0202', 'theta_MV0202', 'V_LV0203', 'theta_LV0203', 'V_MV0203', 'theta_MV0203', 'V_LV0204', 'theta_LV0204', 'V_MV0204', 'theta_MV0204', 'V_LV0205', 'theta_LV0205', 'V_MV0205', 'theta_MV0205', 'V_LV0206', 'theta_LV0206', 'V_MV0206', 'theta_MV0206', 'V_LV0207', 'theta_LV0207', 'V_MV0207', 'theta_MV0207', 'V_LV0208', 'theta_LV0208', 'V_MV0208', 'theta_MV0208', 'V_LV0209', 'theta_LV0209', 'V_MV0209', 'theta_MV0209', 'V_LV0210', 'theta_LV0210', 'V_MV0210', 'theta_MV0210', 'V_LV0211', 'theta_LV0211', 'V_MV0211', 'theta_MV0211', 'V_LV0212', 'theta_LV0212', 'V_MV0212', 'theta_MV0212', 'V_LV0213', 'theta_LV0213', 'V_MV0213', 'theta_MV0213', 'V_LV0214', 'theta_LV0214', 'V_MV0214', 'theta_MV0214', 'V_LV0215', 'theta_LV0215', 'V_MV0215', 'theta_MV0215', 'V_LV0216', 'theta_LV0216', 'V_MV0216', 'theta_MV0216', 'V_LV0217', 'theta_LV0217', 'V_MV0217', 'theta_MV0217', 'V_LV0218', 'theta_LV0218', 'V_MV0218', 'theta_MV0218', 'V_LV0219', 'theta_LV0219', 'V_MV0219', 'theta_MV0219', 'V_LV0220', 'theta_LV0220', 'V_MV0220', 'theta_MV0220', 'p_dc_BESS', 'i_dc_BESS', 'v_dc_BESS', 'omega_GRID', 'i_d_GRID', 'i_q_GRID', 'p_s_GRID', 'q_s_GRID', 'v_dc_LV0101', 'i_sq_ref_LV0101', 'i_sd_ref_LV0101', 'i_sr_LV0101', 'i_si_LV0101', 'p_s_LV0101', 'q_s_LV0101', 'v_dc_LV0102', 'i_sq_ref_LV0102', 'i_sd_ref_LV0102', 'i_sr_LV0102', 'i_si_LV0102', 'p_s_LV0102', 'q_s_LV0102', 'v_dc_LV0103', 'i_sq_ref_LV0103', 'i_sd_ref_LV0103', 'i_sr_LV0103', 'i_si_LV0103', 'p_s_LV0103', 'q_s_LV0103', 'v_dc_LV0104', 'i_sq_ref_LV0104', 'i_sd_ref_LV0104', 'i_sr_LV0104', 'i_si_LV0104', 'p_s_LV0104', 'q_s_LV0104', 'v_dc_LV0105', 'i_sq_ref_LV0105', 'i_sd_ref_LV0105', 'i_sr_LV0105', 'i_si_LV0105', 'p_s_LV0105', 'q_s_LV0105', 'v_dc_LV0106', 'i_sq_ref_LV0106', 'i_sd_ref_LV0106', 'i_sr_LV0106', 'i_si_LV0106', 'p_s_LV0106', 'q_s_LV0106', 'v_dc_LV0107', 'i_sq_ref_LV0107', 'i_sd_ref_LV0107', 'i_sr_LV0107', 'i_si_LV0107', 'p_s_LV0107', 'q_s_LV0107', 'v_dc_LV0108', 'i_sq_ref_LV0108', 'i_sd_ref_LV0108', 'i_sr_LV0108', 'i_si_LV0108', 'p_s_LV0108', 'q_s_LV0108', 'v_dc_LV0109', 'i_sq_ref_LV0109', 'i_sd_ref_LV0109', 'i_sr_LV0109', 'i_si_LV0109', 'p_s_LV0109', 'q_s_LV0109', 'v_dc_LV0110', 'i_sq_ref_LV0110', 'i_sd_ref_LV0110', 'i_sr_LV0110', 'i_si_LV0110', 'p_s_LV0110', 'q_s_LV0110', 'v_dc_LV0111', 'i_sq_ref_LV0111', 'i_sd_ref_LV0111', 'i_sr_LV0111', 'i_si_LV0111', 'p_s_LV0111', 'q_s_LV0111', 'v_dc_LV0112', 'i_sq_ref_LV0112', 'i_sd_ref_LV0112', 'i_sr_LV0112', 'i_si_LV0112', 'p_s_LV0112', 'q_s_LV0112', 'v_dc_LV0113', 'i_sq_ref_LV0113', 'i_sd_ref_LV0113', 'i_sr_LV0113', 'i_si_LV0113', 'p_s_LV0113', 'q_s_LV0113', 'v_dc_LV0114', 'i_sq_ref_LV0114', 'i_sd_ref_LV0114', 'i_sr_LV0114', 'i_si_LV0114', 'p_s_LV0114', 'q_s_LV0114', 'v_dc_LV0115', 'i_sq_ref_LV0115', 'i_sd_ref_LV0115', 'i_sr_LV0115', 'i_si_LV0115', 'p_s_LV0115', 'q_s_LV0115', 'v_dc_LV0116', 'i_sq_ref_LV0116', 'i_sd_ref_LV0116', 'i_sr_LV0116', 'i_si_LV0116', 'p_s_LV0116', 'q_s_LV0116', 'v_dc_LV0117', 'i_sq_ref_LV0117', 'i_sd_ref_LV0117', 'i_sr_LV0117', 'i_si_LV0117', 'p_s_LV0117', 'q_s_LV0117', 'v_dc_LV0118', 'i_sq_ref_LV0118', 'i_sd_ref_LV0118', 'i_sr_LV0118', 'i_si_LV0118', 'p_s_LV0118', 'q_s_LV0118', 'v_dc_LV0119', 'i_sq_ref_LV0119', 'i_sd_ref_LV0119', 'i_sr_LV0119', 'i_si_LV0119', 'p_s_LV0119', 'q_s_LV0119', 'v_dc_LV0120', 'i_sq_ref_LV0120', 'i_sd_ref_LV0120', 'i_sr_LV0120', 'i_si_LV0120', 'p_s_LV0120', 'q_s_LV0120', 'v_dc_LV0201', 'i_sq_ref_LV0201', 'i_sd_ref_LV0201', 'i_sr_LV0201', 'i_si_LV0201', 'p_s_LV0201', 'q_s_LV0201', 'v_dc_LV0202', 'i_sq_ref_LV0202', 'i_sd_ref_LV0202', 'i_sr_LV0202', 'i_si_LV0202', 'p_s_LV0202', 'q_s_LV0202', 'v_dc_LV0203', 'i_sq_ref_LV0203', 'i_sd_ref_LV0203', 'i_sr_LV0203', 'i_si_LV0203', 'p_s_LV0203', 'q_s_LV0203', 'v_dc_LV0204', 'i_sq_ref_LV0204', 'i_sd_ref_LV0204', 'i_sr_LV0204', 'i_si_LV0204', 'p_s_LV0204', 'q_s_LV0204', 'v_dc_LV0205', 'i_sq_ref_LV0205', 'i_sd_ref_LV0205', 'i_sr_LV0205', 'i_si_LV0205', 'p_s_LV0205', 'q_s_LV0205', 'v_dc_LV0206', 'i_sq_ref_LV0206', 'i_sd_ref_LV0206', 'i_sr_LV0206', 'i_si_LV0206', 'p_s_LV0206', 'q_s_LV0206', 'v_dc_LV0207', 'i_sq_ref_LV0207', 'i_sd_ref_LV0207', 'i_sr_LV0207', 'i_si_LV0207', 'p_s_LV0207', 'q_s_LV0207', 'v_dc_LV0208', 'i_sq_ref_LV0208', 'i_sd_ref_LV0208', 'i_sr_LV0208', 'i_si_LV0208', 'p_s_LV0208', 'q_s_LV0208', 'v_dc_LV0209', 'i_sq_ref_LV0209', 'i_sd_ref_LV0209', 'i_sr_LV0209', 'i_si_LV0209', 'p_s_LV0209', 'q_s_LV0209', 'v_dc_LV0210', 'i_sq_ref_LV0210', 'i_sd_ref_LV0210', 'i_sr_LV0210', 'i_si_LV0210', 'p_s_LV0210', 'q_s_LV0210', 'v_dc_LV0211', 'i_sq_ref_LV0211', 'i_sd_ref_LV0211', 'i_sr_LV0211', 'i_si_LV0211', 'p_s_LV0211', 'q_s_LV0211', 'v_dc_LV0212', 'i_sq_ref_LV0212', 'i_sd_ref_LV0212', 'i_sr_LV0212', 'i_si_LV0212', 'p_s_LV0212', 'q_s_LV0212', 'v_dc_LV0213', 'i_sq_ref_LV0213', 'i_sd_ref_LV0213', 'i_sr_LV0213', 'i_si_LV0213', 'p_s_LV0213', 'q_s_LV0213', 'v_dc_LV0214', 'i_sq_ref_LV0214', 'i_sd_ref_LV0214', 'i_sr_LV0214', 'i_si_LV0214', 'p_s_LV0214', 'q_s_LV0214', 'v_dc_LV0215', 'i_sq_ref_LV0215', 'i_sd_ref_LV0215', 'i_sr_LV0215', 'i_si_LV0215', 'p_s_LV0215', 'q_s_LV0215', 'v_dc_LV0216', 'i_sq_ref_LV0216', 'i_sd_ref_LV0216', 'i_sr_LV0216', 'i_si_LV0216', 'p_s_LV0216', 'q_s_LV0216', 'v_dc_LV0217', 'i_sq_ref_LV0217', 'i_sd_ref_LV0217', 'i_sr_LV0217', 'i_si_LV0217', 'p_s_LV0217', 'q_s_LV0217', 'v_dc_LV0218', 'i_sq_ref_LV0218', 'i_sd_ref_LV0218', 'i_sr_LV0218', 'i_si_LV0218', 'p_s_LV0218', 'q_s_LV0218', 'v_dc_LV0219', 'i_sq_ref_LV0219', 'i_sd_ref_LV0219', 'i_sr_LV0219', 'i_si_LV0219', 'p_s_LV0219', 'q_s_LV0219', 'v_dc_LV0220', 'i_sq_ref_LV0220', 'i_sd_ref_LV0220', 'i_sr_LV0220', 'i_si_LV0220', 'p_s_LV0220', 'q_s_LV0220', 'omega_coi', 'p_agc'] 
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
            fobj = BytesIO(pkgutil.get_data(__name__, f'./pv_2_20_sp_jac_ini_num.npz'))
            self.sp_jac_ini = sspa.load_npz(fobj)
        else:
            self.sp_jac_ini = sspa.load_npz(f'{self.matrices_folder}/pv_2_20_sp_jac_ini_num.npz')
            
            
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
            fobj = BytesIO(pkgutil.get_data(__name__, './pv_2_20_sp_jac_run_num.npz'))
            self.sp_jac_run = sspa.load_npz(fobj)
        else:
            self.sp_jac_run = sspa.load_npz(f'{self.matrices_folder}/pv_2_20_sp_jac_run_num.npz')
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
            fobj = BytesIO(pkgutil.get_data(__name__, './pv_2_20_sp_jac_trap_num.npz'))
            self.sp_jac_trap = sspa.load_npz(fobj)
        else:
            self.sp_jac_trap = sspa.load_npz(f'{self.matrices_folder}/pv_2_20_sp_jac_trap_num.npz')
            

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

        self.sp_Fu_run = sspa.load_npz(f'{self.matrices_folder}/pv_2_20_Fu_run_num.npz')
        self.sp_Gu_run = sspa.load_npz(f'{self.matrices_folder}/pv_2_20_Gu_run_num.npz')
        self.sp_Hx_run = sspa.load_npz(f'{self.matrices_folder}/pv_2_20_Hx_run_num.npz')
        self.sp_Hy_run = sspa.load_npz(f'{self.matrices_folder}/pv_2_20_Hy_run_num.npz')
        self.sp_Hu_run = sspa.load_npz(f'{self.matrices_folder}/pv_2_20_Hu_run_num.npz')        
        
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

    sp_jac_ini_ia = [0, 335, 0, 2, 337, 622, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 13, 14, 15, 15, 16, 17, 17, 18, 19, 19, 20, 21, 21, 22, 23, 23, 24, 25, 25, 26, 27, 27, 28, 29, 29, 30, 31, 31, 32, 33, 33, 34, 35, 35, 36, 37, 37, 38, 39, 39, 40, 41, 41, 42, 43, 43, 44, 45, 45, 46, 47, 47, 48, 49, 49, 50, 51, 51, 52, 53, 53, 54, 55, 55, 56, 57, 57, 58, 59, 59, 60, 61, 61, 62, 63, 63, 64, 65, 65, 66, 67, 67, 68, 69, 69, 70, 71, 71, 72, 73, 73, 74, 75, 75, 76, 77, 77, 78, 79, 79, 80, 81, 81, 82, 83, 83, 84, 85, 85, 86, 87, 87, 88, 89, 89, 90, 91, 91, 92, 93, 93, 94, 95, 95, 96, 97, 97, 98, 99, 99, 100, 101, 101, 102, 103, 103, 104, 105, 105, 106, 107, 107, 108, 109, 109, 110, 111, 111, 112, 113, 113, 114, 115, 115, 116, 117, 117, 118, 119, 119, 120, 121, 121, 122, 123, 123, 124, 125, 125, 126, 127, 127, 128, 129, 129, 130, 131, 131, 132, 133, 133, 134, 135, 135, 136, 137, 137, 138, 139, 139, 140, 141, 141, 142, 143, 143, 144, 145, 145, 146, 147, 147, 148, 149, 149, 150, 151, 151, 152, 153, 153, 154, 155, 155, 156, 157, 157, 158, 159, 159, 160, 161, 161, 162, 163, 163, 164, 165, 622, 166, 167, 168, 169, 172, 173, 176, 177, 256, 257, 166, 167, 168, 169, 172, 173, 176, 177, 256, 257, 166, 167, 168, 169, 170, 171, 166, 167, 168, 169, 170, 171, 168, 169, 170, 171, 340, 168, 169, 170, 171, 341, 0, 1, 166, 167, 172, 173, 166, 167, 172, 173, 174, 175, 176, 177, 347, 174, 175, 176, 177, 348, 166, 167, 174, 175, 176, 177, 180, 181, 166, 167, 174, 175, 176, 177, 180, 181, 178, 179, 180, 181, 354, 178, 179, 180, 181, 355, 176, 177, 178, 179, 180, 181, 184, 185, 176, 177, 178, 179, 180, 181, 184, 185, 182, 183, 184, 185, 361, 182, 183, 184, 185, 362, 180, 181, 182, 183, 184, 185, 188, 189, 180, 181, 182, 183, 184, 185, 188, 189, 186, 187, 188, 189, 368, 186, 187, 188, 189, 369, 184, 185, 186, 187, 188, 189, 192, 193, 184, 185, 186, 187, 188, 189, 192, 193, 190, 191, 192, 193, 375, 190, 191, 192, 193, 376, 188, 189, 190, 191, 192, 193, 196, 197, 188, 189, 190, 191, 192, 193, 196, 197, 194, 195, 196, 197, 382, 194, 195, 196, 197, 383, 192, 193, 194, 195, 196, 197, 200, 201, 192, 193, 194, 195, 196, 197, 200, 201, 198, 199, 200, 201, 389, 198, 199, 200, 201, 390, 196, 197, 198, 199, 200, 201, 204, 205, 196, 197, 198, 199, 200, 201, 204, 205, 202, 203, 204, 205, 396, 202, 203, 204, 205, 397, 200, 201, 202, 203, 204, 205, 208, 209, 200, 201, 202, 203, 204, 205, 208, 209, 206, 207, 208, 209, 403, 206, 207, 208, 209, 404, 204, 205, 206, 207, 208, 209, 212, 213, 204, 205, 206, 207, 208, 209, 212, 213, 210, 211, 212, 213, 410, 210, 211, 212, 213, 411, 208, 209, 210, 211, 212, 213, 216, 217, 208, 209, 210, 211, 212, 213, 216, 217, 214, 215, 216, 217, 417, 214, 215, 216, 217, 418, 212, 213, 214, 215, 216, 217, 220, 221, 212, 213, 214, 215, 216, 217, 220, 221, 218, 219, 220, 221, 424, 218, 219, 220, 221, 425, 216, 217, 218, 219, 220, 221, 224, 225, 216, 217, 218, 219, 220, 221, 224, 225, 222, 223, 224, 225, 431, 222, 223, 224, 225, 432, 220, 221, 222, 223, 224, 225, 228, 229, 220, 221, 222, 223, 224, 225, 228, 229, 226, 227, 228, 229, 438, 226, 227, 228, 229, 439, 224, 225, 226, 227, 228, 229, 232, 233, 224, 225, 226, 227, 228, 229, 232, 233, 230, 231, 232, 233, 445, 230, 231, 232, 233, 446, 228, 229, 230, 231, 232, 233, 236, 237, 228, 229, 230, 231, 232, 233, 236, 237, 234, 235, 236, 237, 452, 234, 235, 236, 237, 453, 232, 233, 234, 235, 236, 237, 240, 241, 232, 233, 234, 235, 236, 237, 240, 241, 238, 239, 240, 241, 459, 238, 239, 240, 241, 460, 236, 237, 238, 239, 240, 241, 244, 245, 236, 237, 238, 239, 240, 241, 244, 245, 242, 243, 244, 245, 466, 242, 243, 244, 245, 467, 240, 241, 242, 243, 244, 245, 248, 249, 240, 241, 242, 243, 244, 245, 248, 249, 246, 247, 248, 249, 473, 246, 247, 248, 249, 474, 244, 245, 246, 247, 248, 249, 252, 253, 244, 245, 246, 247, 248, 249, 252, 253, 250, 251, 252, 253, 480, 250, 251, 252, 253, 481, 248, 249, 250, 251, 252, 253, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 487, 254, 255, 256, 257, 488, 166, 167, 254, 255, 256, 257, 260, 261, 166, 167, 254, 255, 256, 257, 260, 261, 258, 259, 260, 261, 494, 258, 259, 260, 261, 495, 256, 257, 258, 259, 260, 261, 264, 265, 256, 257, 258, 259, 260, 261, 264, 265, 262, 263, 264, 265, 501, 262, 263, 264, 265, 502, 260, 261, 262, 263, 264, 265, 268, 269, 260, 261, 262, 263, 264, 265, 268, 269, 266, 267, 268, 269, 508, 266, 267, 268, 269, 509, 264, 265, 266, 267, 268, 269, 272, 273, 264, 265, 266, 267, 268, 269, 272, 273, 270, 271, 272, 273, 515, 270, 271, 272, 273, 516, 268, 269, 270, 271, 272, 273, 276, 277, 268, 269, 270, 271, 272, 273, 276, 277, 274, 275, 276, 277, 522, 274, 275, 276, 277, 523, 272, 273, 274, 275, 276, 277, 280, 281, 272, 273, 274, 275, 276, 277, 280, 281, 278, 279, 280, 281, 529, 278, 279, 280, 281, 530, 276, 277, 278, 279, 280, 281, 284, 285, 276, 277, 278, 279, 280, 281, 284, 285, 282, 283, 284, 285, 536, 282, 283, 284, 285, 537, 280, 281, 282, 283, 284, 285, 288, 289, 280, 281, 282, 283, 284, 285, 288, 289, 286, 287, 288, 289, 543, 286, 287, 288, 289, 544, 284, 285, 286, 287, 288, 289, 292, 293, 284, 285, 286, 287, 288, 289, 292, 293, 290, 291, 292, 293, 550, 290, 291, 292, 293, 551, 288, 289, 290, 291, 292, 293, 296, 297, 288, 289, 290, 291, 292, 293, 296, 297, 294, 295, 296, 297, 557, 294, 295, 296, 297, 558, 292, 293, 294, 295, 296, 297, 300, 301, 292, 293, 294, 295, 296, 297, 300, 301, 298, 299, 300, 301, 564, 298, 299, 300, 301, 565, 296, 297, 298, 299, 300, 301, 304, 305, 296, 297, 298, 299, 300, 301, 304, 305, 302, 303, 304, 305, 571, 302, 303, 304, 305, 572, 300, 301, 302, 303, 304, 305, 308, 309, 300, 301, 302, 303, 304, 305, 308, 309, 306, 307, 308, 309, 578, 306, 307, 308, 309, 579, 304, 305, 306, 307, 308, 309, 312, 313, 304, 305, 306, 307, 308, 309, 312, 313, 310, 311, 312, 313, 585, 310, 311, 312, 313, 586, 308, 309, 310, 311, 312, 313, 316, 317, 308, 309, 310, 311, 312, 313, 316, 317, 314, 315, 316, 317, 592, 314, 315, 316, 317, 593, 312, 313, 314, 315, 316, 317, 320, 321, 312, 313, 314, 315, 316, 317, 320, 321, 318, 319, 320, 321, 599, 318, 319, 320, 321, 600, 316, 317, 318, 319, 320, 321, 324, 325, 316, 317, 318, 319, 320, 321, 324, 325, 322, 323, 324, 325, 606, 322, 323, 324, 325, 607, 320, 321, 322, 323, 324, 325, 328, 329, 320, 321, 322, 323, 324, 325, 328, 329, 326, 327, 328, 329, 613, 326, 327, 328, 329, 614, 324, 325, 326, 327, 328, 329, 332, 333, 324, 325, 326, 327, 328, 329, 332, 333, 330, 331, 332, 333, 620, 330, 331, 332, 333, 621, 328, 329, 330, 331, 332, 333, 328, 329, 330, 331, 332, 333, 0, 1, 172, 334, 334, 335, 336, 0, 335, 336, 3, 337, 2, 170, 171, 338, 339, 2, 4, 170, 171, 338, 339, 2, 170, 171, 338, 339, 340, 2, 170, 171, 338, 339, 341, 342, 347, 8, 174, 344, 6, 174, 343, 174, 175, 343, 344, 345, 346, 174, 175, 343, 344, 345, 346, 174, 175, 345, 346, 347, 174, 175, 345, 346, 348, 349, 354, 12, 178, 351, 10, 178, 350, 178, 179, 350, 351, 352, 353, 178, 179, 350, 351, 352, 353, 178, 179, 352, 353, 354, 178, 179, 352, 353, 355, 356, 361, 16, 182, 358, 14, 182, 357, 182, 183, 357, 358, 359, 360, 182, 183, 357, 358, 359, 360, 182, 183, 359, 360, 361, 182, 183, 359, 360, 362, 363, 368, 20, 186, 365, 18, 186, 364, 186, 187, 364, 365, 366, 367, 186, 187, 364, 365, 366, 367, 186, 187, 366, 367, 368, 186, 187, 366, 367, 369, 370, 375, 24, 190, 372, 22, 190, 371, 190, 191, 371, 372, 373, 374, 190, 191, 371, 372, 373, 374, 190, 191, 373, 374, 375, 190, 191, 373, 374, 376, 377, 382, 28, 194, 379, 26, 194, 378, 194, 195, 378, 379, 380, 381, 194, 195, 378, 379, 380, 381, 194, 195, 380, 381, 382, 194, 195, 380, 381, 383, 384, 389, 32, 198, 386, 30, 198, 385, 198, 199, 385, 386, 387, 388, 198, 199, 385, 386, 387, 388, 198, 199, 387, 388, 389, 198, 199, 387, 388, 390, 391, 396, 36, 202, 393, 34, 202, 392, 202, 203, 392, 393, 394, 395, 202, 203, 392, 393, 394, 395, 202, 203, 394, 395, 396, 202, 203, 394, 395, 397, 398, 403, 40, 206, 400, 38, 206, 399, 206, 207, 399, 400, 401, 402, 206, 207, 399, 400, 401, 402, 206, 207, 401, 402, 403, 206, 207, 401, 402, 404, 405, 410, 44, 210, 407, 42, 210, 406, 210, 211, 406, 407, 408, 409, 210, 211, 406, 407, 408, 409, 210, 211, 408, 409, 410, 210, 211, 408, 409, 411, 412, 417, 48, 214, 414, 46, 214, 413, 214, 215, 413, 414, 415, 416, 214, 215, 413, 414, 415, 416, 214, 215, 415, 416, 417, 214, 215, 415, 416, 418, 419, 424, 52, 218, 421, 50, 218, 420, 218, 219, 420, 421, 422, 423, 218, 219, 420, 421, 422, 423, 218, 219, 422, 423, 424, 218, 219, 422, 423, 425, 426, 431, 56, 222, 428, 54, 222, 427, 222, 223, 427, 428, 429, 430, 222, 223, 427, 428, 429, 430, 222, 223, 429, 430, 431, 222, 223, 429, 430, 432, 433, 438, 60, 226, 435, 58, 226, 434, 226, 227, 434, 435, 436, 437, 226, 227, 434, 435, 436, 437, 226, 227, 436, 437, 438, 226, 227, 436, 437, 439, 440, 445, 64, 230, 442, 62, 230, 441, 230, 231, 441, 442, 443, 444, 230, 231, 441, 442, 443, 444, 230, 231, 443, 444, 445, 230, 231, 443, 444, 446, 447, 452, 68, 234, 449, 66, 234, 448, 234, 235, 448, 449, 450, 451, 234, 235, 448, 449, 450, 451, 234, 235, 450, 451, 452, 234, 235, 450, 451, 453, 454, 459, 72, 238, 456, 70, 238, 455, 238, 239, 455, 456, 457, 458, 238, 239, 455, 456, 457, 458, 238, 239, 457, 458, 459, 238, 239, 457, 458, 460, 461, 466, 76, 242, 463, 74, 242, 462, 242, 243, 462, 463, 464, 465, 242, 243, 462, 463, 464, 465, 242, 243, 464, 465, 466, 242, 243, 464, 465, 467, 468, 473, 80, 246, 470, 78, 246, 469, 246, 247, 469, 470, 471, 472, 246, 247, 469, 470, 471, 472, 246, 247, 471, 472, 473, 246, 247, 471, 472, 474, 475, 480, 84, 250, 477, 82, 250, 476, 250, 251, 476, 477, 478, 479, 250, 251, 476, 477, 478, 479, 250, 251, 478, 479, 480, 250, 251, 478, 479, 481, 482, 487, 88, 254, 484, 86, 254, 483, 254, 255, 483, 484, 485, 486, 254, 255, 483, 484, 485, 486, 254, 255, 485, 486, 487, 254, 255, 485, 486, 488, 489, 494, 92, 258, 491, 90, 258, 490, 258, 259, 490, 491, 492, 493, 258, 259, 490, 491, 492, 493, 258, 259, 492, 493, 494, 258, 259, 492, 493, 495, 496, 501, 96, 262, 498, 94, 262, 497, 262, 263, 497, 498, 499, 500, 262, 263, 497, 498, 499, 500, 262, 263, 499, 500, 501, 262, 263, 499, 500, 502, 503, 508, 100, 266, 505, 98, 266, 504, 266, 267, 504, 505, 506, 507, 266, 267, 504, 505, 506, 507, 266, 267, 506, 507, 508, 266, 267, 506, 507, 509, 510, 515, 104, 270, 512, 102, 270, 511, 270, 271, 511, 512, 513, 514, 270, 271, 511, 512, 513, 514, 270, 271, 513, 514, 515, 270, 271, 513, 514, 516, 517, 522, 108, 274, 519, 106, 274, 518, 274, 275, 518, 519, 520, 521, 274, 275, 518, 519, 520, 521, 274, 275, 520, 521, 522, 274, 275, 520, 521, 523, 524, 529, 112, 278, 526, 110, 278, 525, 278, 279, 525, 526, 527, 528, 278, 279, 525, 526, 527, 528, 278, 279, 527, 528, 529, 278, 279, 527, 528, 530, 531, 536, 116, 282, 533, 114, 282, 532, 282, 283, 532, 533, 534, 535, 282, 283, 532, 533, 534, 535, 282, 283, 534, 535, 536, 282, 283, 534, 535, 537, 538, 543, 120, 286, 540, 118, 286, 539, 286, 287, 539, 540, 541, 542, 286, 287, 539, 540, 541, 542, 286, 287, 541, 542, 543, 286, 287, 541, 542, 544, 545, 550, 124, 290, 547, 122, 290, 546, 290, 291, 546, 547, 548, 549, 290, 291, 546, 547, 548, 549, 290, 291, 548, 549, 550, 290, 291, 548, 549, 551, 552, 557, 128, 294, 554, 126, 294, 553, 294, 295, 553, 554, 555, 556, 294, 295, 553, 554, 555, 556, 294, 295, 555, 556, 557, 294, 295, 555, 556, 558, 559, 564, 132, 298, 561, 130, 298, 560, 298, 299, 560, 561, 562, 563, 298, 299, 560, 561, 562, 563, 298, 299, 562, 563, 564, 298, 299, 562, 563, 565, 566, 571, 136, 302, 568, 134, 302, 567, 302, 303, 567, 568, 569, 570, 302, 303, 567, 568, 569, 570, 302, 303, 569, 570, 571, 302, 303, 569, 570, 572, 573, 578, 140, 306, 575, 138, 306, 574, 306, 307, 574, 575, 576, 577, 306, 307, 574, 575, 576, 577, 306, 307, 576, 577, 578, 306, 307, 576, 577, 579, 580, 585, 144, 310, 582, 142, 310, 581, 310, 311, 581, 582, 583, 584, 310, 311, 581, 582, 583, 584, 310, 311, 583, 584, 585, 310, 311, 583, 584, 586, 587, 592, 148, 314, 589, 146, 314, 588, 314, 315, 588, 589, 590, 591, 314, 315, 588, 589, 590, 591, 314, 315, 590, 591, 592, 314, 315, 590, 591, 593, 594, 599, 152, 318, 596, 150, 318, 595, 318, 319, 595, 596, 597, 598, 318, 319, 595, 596, 597, 598, 318, 319, 597, 598, 599, 318, 319, 597, 598, 600, 601, 606, 156, 322, 603, 154, 322, 602, 322, 323, 602, 603, 604, 605, 322, 323, 602, 603, 604, 605, 322, 323, 604, 605, 606, 322, 323, 604, 605, 607, 608, 613, 160, 326, 610, 158, 326, 609, 326, 327, 609, 610, 611, 612, 326, 327, 609, 610, 611, 612, 326, 327, 611, 612, 613, 326, 327, 611, 612, 614, 615, 620, 164, 330, 617, 162, 330, 616, 330, 331, 616, 617, 618, 619, 330, 331, 616, 617, 618, 619, 330, 331, 618, 619, 620, 330, 331, 618, 619, 621, 337, 622, 165, 622, 623]
    sp_jac_ini_ja = [0, 2, 3, 6, 7, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29, 30, 32, 33, 35, 36, 38, 39, 41, 42, 44, 45, 47, 48, 50, 51, 53, 54, 56, 57, 59, 60, 62, 63, 65, 66, 68, 69, 71, 72, 74, 75, 77, 78, 80, 81, 83, 84, 86, 87, 89, 90, 92, 93, 95, 96, 98, 99, 101, 102, 104, 105, 107, 108, 110, 111, 113, 114, 116, 117, 119, 120, 122, 123, 125, 126, 128, 129, 131, 132, 134, 135, 137, 138, 140, 141, 143, 144, 146, 147, 149, 150, 152, 153, 155, 156, 158, 159, 161, 162, 164, 165, 167, 168, 170, 171, 173, 174, 176, 177, 179, 180, 182, 183, 185, 186, 188, 189, 191, 192, 194, 195, 197, 198, 200, 201, 203, 204, 206, 207, 209, 210, 212, 213, 215, 216, 218, 219, 221, 222, 224, 225, 227, 228, 230, 231, 233, 234, 236, 237, 239, 240, 242, 243, 245, 246, 248, 250, 260, 270, 276, 282, 287, 292, 298, 302, 307, 312, 320, 328, 333, 338, 346, 354, 359, 364, 372, 380, 385, 390, 398, 406, 411, 416, 424, 432, 437, 442, 450, 458, 463, 468, 476, 484, 489, 494, 502, 510, 515, 520, 528, 536, 541, 546, 554, 562, 567, 572, 580, 588, 593, 598, 606, 614, 619, 624, 632, 640, 645, 650, 658, 666, 671, 676, 684, 692, 697, 702, 710, 718, 723, 728, 736, 744, 749, 754, 762, 770, 775, 780, 788, 796, 801, 806, 812, 818, 823, 828, 836, 844, 849, 854, 862, 870, 875, 880, 888, 896, 901, 906, 914, 922, 927, 932, 940, 948, 953, 958, 966, 974, 979, 984, 992, 1000, 1005, 1010, 1018, 1026, 1031, 1036, 1044, 1052, 1057, 1062, 1070, 1078, 1083, 1088, 1096, 1104, 1109, 1114, 1122, 1130, 1135, 1140, 1148, 1156, 1161, 1166, 1174, 1182, 1187, 1192, 1200, 1208, 1213, 1218, 1226, 1234, 1239, 1244, 1252, 1260, 1265, 1270, 1278, 1286, 1291, 1296, 1304, 1312, 1317, 1322, 1328, 1334, 1338, 1341, 1344, 1346, 1351, 1357, 1363, 1369, 1371, 1374, 1377, 1383, 1389, 1394, 1399, 1401, 1404, 1407, 1413, 1419, 1424, 1429, 1431, 1434, 1437, 1443, 1449, 1454, 1459, 1461, 1464, 1467, 1473, 1479, 1484, 1489, 1491, 1494, 1497, 1503, 1509, 1514, 1519, 1521, 1524, 1527, 1533, 1539, 1544, 1549, 1551, 1554, 1557, 1563, 1569, 1574, 1579, 1581, 1584, 1587, 1593, 1599, 1604, 1609, 1611, 1614, 1617, 1623, 1629, 1634, 1639, 1641, 1644, 1647, 1653, 1659, 1664, 1669, 1671, 1674, 1677, 1683, 1689, 1694, 1699, 1701, 1704, 1707, 1713, 1719, 1724, 1729, 1731, 1734, 1737, 1743, 1749, 1754, 1759, 1761, 1764, 1767, 1773, 1779, 1784, 1789, 1791, 1794, 1797, 1803, 1809, 1814, 1819, 1821, 1824, 1827, 1833, 1839, 1844, 1849, 1851, 1854, 1857, 1863, 1869, 1874, 1879, 1881, 1884, 1887, 1893, 1899, 1904, 1909, 1911, 1914, 1917, 1923, 1929, 1934, 1939, 1941, 1944, 1947, 1953, 1959, 1964, 1969, 1971, 1974, 1977, 1983, 1989, 1994, 1999, 2001, 2004, 2007, 2013, 2019, 2024, 2029, 2031, 2034, 2037, 2043, 2049, 2054, 2059, 2061, 2064, 2067, 2073, 2079, 2084, 2089, 2091, 2094, 2097, 2103, 2109, 2114, 2119, 2121, 2124, 2127, 2133, 2139, 2144, 2149, 2151, 2154, 2157, 2163, 2169, 2174, 2179, 2181, 2184, 2187, 2193, 2199, 2204, 2209, 2211, 2214, 2217, 2223, 2229, 2234, 2239, 2241, 2244, 2247, 2253, 2259, 2264, 2269, 2271, 2274, 2277, 2283, 2289, 2294, 2299, 2301, 2304, 2307, 2313, 2319, 2324, 2329, 2331, 2334, 2337, 2343, 2349, 2354, 2359, 2361, 2364, 2367, 2373, 2379, 2384, 2389, 2391, 2394, 2397, 2403, 2409, 2414, 2419, 2421, 2424, 2427, 2433, 2439, 2444, 2449, 2451, 2454, 2457, 2463, 2469, 2474, 2479, 2481, 2484, 2487, 2493, 2499, 2504, 2509, 2511, 2514, 2517, 2523, 2529, 2534, 2539, 2541, 2544, 2547, 2553, 2559, 2564, 2569, 2571, 2574]
    sp_jac_ini_nia = 624
    sp_jac_ini_nja = 624
    return sp_jac_ini_ia, sp_jac_ini_ja, sp_jac_ini_nia, sp_jac_ini_nja 

def sp_jac_run_vectors():

    sp_jac_run_ia = [0, 335, 0, 2, 337, 622, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 13, 14, 15, 15, 16, 17, 17, 18, 19, 19, 20, 21, 21, 22, 23, 23, 24, 25, 25, 26, 27, 27, 28, 29, 29, 30, 31, 31, 32, 33, 33, 34, 35, 35, 36, 37, 37, 38, 39, 39, 40, 41, 41, 42, 43, 43, 44, 45, 45, 46, 47, 47, 48, 49, 49, 50, 51, 51, 52, 53, 53, 54, 55, 55, 56, 57, 57, 58, 59, 59, 60, 61, 61, 62, 63, 63, 64, 65, 65, 66, 67, 67, 68, 69, 69, 70, 71, 71, 72, 73, 73, 74, 75, 75, 76, 77, 77, 78, 79, 79, 80, 81, 81, 82, 83, 83, 84, 85, 85, 86, 87, 87, 88, 89, 89, 90, 91, 91, 92, 93, 93, 94, 95, 95, 96, 97, 97, 98, 99, 99, 100, 101, 101, 102, 103, 103, 104, 105, 105, 106, 107, 107, 108, 109, 109, 110, 111, 111, 112, 113, 113, 114, 115, 115, 116, 117, 117, 118, 119, 119, 120, 121, 121, 122, 123, 123, 124, 125, 125, 126, 127, 127, 128, 129, 129, 130, 131, 131, 132, 133, 133, 134, 135, 135, 136, 137, 137, 138, 139, 139, 140, 141, 141, 142, 143, 143, 144, 145, 145, 146, 147, 147, 148, 149, 149, 150, 151, 151, 152, 153, 153, 154, 155, 155, 156, 157, 157, 158, 159, 159, 160, 161, 161, 162, 163, 163, 164, 165, 622, 166, 167, 168, 169, 172, 173, 176, 177, 256, 257, 166, 167, 168, 169, 172, 173, 176, 177, 256, 257, 166, 167, 168, 169, 170, 171, 166, 167, 168, 169, 170, 171, 168, 169, 170, 171, 340, 168, 169, 170, 171, 341, 0, 1, 166, 167, 172, 173, 166, 167, 172, 173, 174, 175, 176, 177, 347, 174, 175, 176, 177, 348, 166, 167, 174, 175, 176, 177, 180, 181, 166, 167, 174, 175, 176, 177, 180, 181, 178, 179, 180, 181, 354, 178, 179, 180, 181, 355, 176, 177, 178, 179, 180, 181, 184, 185, 176, 177, 178, 179, 180, 181, 184, 185, 182, 183, 184, 185, 361, 182, 183, 184, 185, 362, 180, 181, 182, 183, 184, 185, 188, 189, 180, 181, 182, 183, 184, 185, 188, 189, 186, 187, 188, 189, 368, 186, 187, 188, 189, 369, 184, 185, 186, 187, 188, 189, 192, 193, 184, 185, 186, 187, 188, 189, 192, 193, 190, 191, 192, 193, 375, 190, 191, 192, 193, 376, 188, 189, 190, 191, 192, 193, 196, 197, 188, 189, 190, 191, 192, 193, 196, 197, 194, 195, 196, 197, 382, 194, 195, 196, 197, 383, 192, 193, 194, 195, 196, 197, 200, 201, 192, 193, 194, 195, 196, 197, 200, 201, 198, 199, 200, 201, 389, 198, 199, 200, 201, 390, 196, 197, 198, 199, 200, 201, 204, 205, 196, 197, 198, 199, 200, 201, 204, 205, 202, 203, 204, 205, 396, 202, 203, 204, 205, 397, 200, 201, 202, 203, 204, 205, 208, 209, 200, 201, 202, 203, 204, 205, 208, 209, 206, 207, 208, 209, 403, 206, 207, 208, 209, 404, 204, 205, 206, 207, 208, 209, 212, 213, 204, 205, 206, 207, 208, 209, 212, 213, 210, 211, 212, 213, 410, 210, 211, 212, 213, 411, 208, 209, 210, 211, 212, 213, 216, 217, 208, 209, 210, 211, 212, 213, 216, 217, 214, 215, 216, 217, 417, 214, 215, 216, 217, 418, 212, 213, 214, 215, 216, 217, 220, 221, 212, 213, 214, 215, 216, 217, 220, 221, 218, 219, 220, 221, 424, 218, 219, 220, 221, 425, 216, 217, 218, 219, 220, 221, 224, 225, 216, 217, 218, 219, 220, 221, 224, 225, 222, 223, 224, 225, 431, 222, 223, 224, 225, 432, 220, 221, 222, 223, 224, 225, 228, 229, 220, 221, 222, 223, 224, 225, 228, 229, 226, 227, 228, 229, 438, 226, 227, 228, 229, 439, 224, 225, 226, 227, 228, 229, 232, 233, 224, 225, 226, 227, 228, 229, 232, 233, 230, 231, 232, 233, 445, 230, 231, 232, 233, 446, 228, 229, 230, 231, 232, 233, 236, 237, 228, 229, 230, 231, 232, 233, 236, 237, 234, 235, 236, 237, 452, 234, 235, 236, 237, 453, 232, 233, 234, 235, 236, 237, 240, 241, 232, 233, 234, 235, 236, 237, 240, 241, 238, 239, 240, 241, 459, 238, 239, 240, 241, 460, 236, 237, 238, 239, 240, 241, 244, 245, 236, 237, 238, 239, 240, 241, 244, 245, 242, 243, 244, 245, 466, 242, 243, 244, 245, 467, 240, 241, 242, 243, 244, 245, 248, 249, 240, 241, 242, 243, 244, 245, 248, 249, 246, 247, 248, 249, 473, 246, 247, 248, 249, 474, 244, 245, 246, 247, 248, 249, 252, 253, 244, 245, 246, 247, 248, 249, 252, 253, 250, 251, 252, 253, 480, 250, 251, 252, 253, 481, 248, 249, 250, 251, 252, 253, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 487, 254, 255, 256, 257, 488, 166, 167, 254, 255, 256, 257, 260, 261, 166, 167, 254, 255, 256, 257, 260, 261, 258, 259, 260, 261, 494, 258, 259, 260, 261, 495, 256, 257, 258, 259, 260, 261, 264, 265, 256, 257, 258, 259, 260, 261, 264, 265, 262, 263, 264, 265, 501, 262, 263, 264, 265, 502, 260, 261, 262, 263, 264, 265, 268, 269, 260, 261, 262, 263, 264, 265, 268, 269, 266, 267, 268, 269, 508, 266, 267, 268, 269, 509, 264, 265, 266, 267, 268, 269, 272, 273, 264, 265, 266, 267, 268, 269, 272, 273, 270, 271, 272, 273, 515, 270, 271, 272, 273, 516, 268, 269, 270, 271, 272, 273, 276, 277, 268, 269, 270, 271, 272, 273, 276, 277, 274, 275, 276, 277, 522, 274, 275, 276, 277, 523, 272, 273, 274, 275, 276, 277, 280, 281, 272, 273, 274, 275, 276, 277, 280, 281, 278, 279, 280, 281, 529, 278, 279, 280, 281, 530, 276, 277, 278, 279, 280, 281, 284, 285, 276, 277, 278, 279, 280, 281, 284, 285, 282, 283, 284, 285, 536, 282, 283, 284, 285, 537, 280, 281, 282, 283, 284, 285, 288, 289, 280, 281, 282, 283, 284, 285, 288, 289, 286, 287, 288, 289, 543, 286, 287, 288, 289, 544, 284, 285, 286, 287, 288, 289, 292, 293, 284, 285, 286, 287, 288, 289, 292, 293, 290, 291, 292, 293, 550, 290, 291, 292, 293, 551, 288, 289, 290, 291, 292, 293, 296, 297, 288, 289, 290, 291, 292, 293, 296, 297, 294, 295, 296, 297, 557, 294, 295, 296, 297, 558, 292, 293, 294, 295, 296, 297, 300, 301, 292, 293, 294, 295, 296, 297, 300, 301, 298, 299, 300, 301, 564, 298, 299, 300, 301, 565, 296, 297, 298, 299, 300, 301, 304, 305, 296, 297, 298, 299, 300, 301, 304, 305, 302, 303, 304, 305, 571, 302, 303, 304, 305, 572, 300, 301, 302, 303, 304, 305, 308, 309, 300, 301, 302, 303, 304, 305, 308, 309, 306, 307, 308, 309, 578, 306, 307, 308, 309, 579, 304, 305, 306, 307, 308, 309, 312, 313, 304, 305, 306, 307, 308, 309, 312, 313, 310, 311, 312, 313, 585, 310, 311, 312, 313, 586, 308, 309, 310, 311, 312, 313, 316, 317, 308, 309, 310, 311, 312, 313, 316, 317, 314, 315, 316, 317, 592, 314, 315, 316, 317, 593, 312, 313, 314, 315, 316, 317, 320, 321, 312, 313, 314, 315, 316, 317, 320, 321, 318, 319, 320, 321, 599, 318, 319, 320, 321, 600, 316, 317, 318, 319, 320, 321, 324, 325, 316, 317, 318, 319, 320, 321, 324, 325, 322, 323, 324, 325, 606, 322, 323, 324, 325, 607, 320, 321, 322, 323, 324, 325, 328, 329, 320, 321, 322, 323, 324, 325, 328, 329, 326, 327, 328, 329, 613, 326, 327, 328, 329, 614, 324, 325, 326, 327, 328, 329, 332, 333, 324, 325, 326, 327, 328, 329, 332, 333, 330, 331, 332, 333, 620, 330, 331, 332, 333, 621, 328, 329, 330, 331, 332, 333, 328, 329, 330, 331, 332, 333, 0, 1, 172, 334, 334, 335, 336, 0, 335, 336, 3, 337, 2, 170, 171, 338, 339, 2, 4, 170, 171, 338, 339, 2, 170, 171, 338, 339, 340, 2, 170, 171, 338, 339, 341, 342, 347, 8, 174, 344, 6, 174, 343, 174, 175, 343, 344, 345, 346, 174, 175, 343, 344, 345, 346, 174, 175, 345, 346, 347, 174, 175, 345, 346, 348, 349, 354, 12, 178, 351, 10, 178, 350, 178, 179, 350, 351, 352, 353, 178, 179, 350, 351, 352, 353, 178, 179, 352, 353, 354, 178, 179, 352, 353, 355, 356, 361, 16, 182, 358, 14, 182, 357, 182, 183, 357, 358, 359, 360, 182, 183, 357, 358, 359, 360, 182, 183, 359, 360, 361, 182, 183, 359, 360, 362, 363, 368, 20, 186, 365, 18, 186, 364, 186, 187, 364, 365, 366, 367, 186, 187, 364, 365, 366, 367, 186, 187, 366, 367, 368, 186, 187, 366, 367, 369, 370, 375, 24, 190, 372, 22, 190, 371, 190, 191, 371, 372, 373, 374, 190, 191, 371, 372, 373, 374, 190, 191, 373, 374, 375, 190, 191, 373, 374, 376, 377, 382, 28, 194, 379, 26, 194, 378, 194, 195, 378, 379, 380, 381, 194, 195, 378, 379, 380, 381, 194, 195, 380, 381, 382, 194, 195, 380, 381, 383, 384, 389, 32, 198, 386, 30, 198, 385, 198, 199, 385, 386, 387, 388, 198, 199, 385, 386, 387, 388, 198, 199, 387, 388, 389, 198, 199, 387, 388, 390, 391, 396, 36, 202, 393, 34, 202, 392, 202, 203, 392, 393, 394, 395, 202, 203, 392, 393, 394, 395, 202, 203, 394, 395, 396, 202, 203, 394, 395, 397, 398, 403, 40, 206, 400, 38, 206, 399, 206, 207, 399, 400, 401, 402, 206, 207, 399, 400, 401, 402, 206, 207, 401, 402, 403, 206, 207, 401, 402, 404, 405, 410, 44, 210, 407, 42, 210, 406, 210, 211, 406, 407, 408, 409, 210, 211, 406, 407, 408, 409, 210, 211, 408, 409, 410, 210, 211, 408, 409, 411, 412, 417, 48, 214, 414, 46, 214, 413, 214, 215, 413, 414, 415, 416, 214, 215, 413, 414, 415, 416, 214, 215, 415, 416, 417, 214, 215, 415, 416, 418, 419, 424, 52, 218, 421, 50, 218, 420, 218, 219, 420, 421, 422, 423, 218, 219, 420, 421, 422, 423, 218, 219, 422, 423, 424, 218, 219, 422, 423, 425, 426, 431, 56, 222, 428, 54, 222, 427, 222, 223, 427, 428, 429, 430, 222, 223, 427, 428, 429, 430, 222, 223, 429, 430, 431, 222, 223, 429, 430, 432, 433, 438, 60, 226, 435, 58, 226, 434, 226, 227, 434, 435, 436, 437, 226, 227, 434, 435, 436, 437, 226, 227, 436, 437, 438, 226, 227, 436, 437, 439, 440, 445, 64, 230, 442, 62, 230, 441, 230, 231, 441, 442, 443, 444, 230, 231, 441, 442, 443, 444, 230, 231, 443, 444, 445, 230, 231, 443, 444, 446, 447, 452, 68, 234, 449, 66, 234, 448, 234, 235, 448, 449, 450, 451, 234, 235, 448, 449, 450, 451, 234, 235, 450, 451, 452, 234, 235, 450, 451, 453, 454, 459, 72, 238, 456, 70, 238, 455, 238, 239, 455, 456, 457, 458, 238, 239, 455, 456, 457, 458, 238, 239, 457, 458, 459, 238, 239, 457, 458, 460, 461, 466, 76, 242, 463, 74, 242, 462, 242, 243, 462, 463, 464, 465, 242, 243, 462, 463, 464, 465, 242, 243, 464, 465, 466, 242, 243, 464, 465, 467, 468, 473, 80, 246, 470, 78, 246, 469, 246, 247, 469, 470, 471, 472, 246, 247, 469, 470, 471, 472, 246, 247, 471, 472, 473, 246, 247, 471, 472, 474, 475, 480, 84, 250, 477, 82, 250, 476, 250, 251, 476, 477, 478, 479, 250, 251, 476, 477, 478, 479, 250, 251, 478, 479, 480, 250, 251, 478, 479, 481, 482, 487, 88, 254, 484, 86, 254, 483, 254, 255, 483, 484, 485, 486, 254, 255, 483, 484, 485, 486, 254, 255, 485, 486, 487, 254, 255, 485, 486, 488, 489, 494, 92, 258, 491, 90, 258, 490, 258, 259, 490, 491, 492, 493, 258, 259, 490, 491, 492, 493, 258, 259, 492, 493, 494, 258, 259, 492, 493, 495, 496, 501, 96, 262, 498, 94, 262, 497, 262, 263, 497, 498, 499, 500, 262, 263, 497, 498, 499, 500, 262, 263, 499, 500, 501, 262, 263, 499, 500, 502, 503, 508, 100, 266, 505, 98, 266, 504, 266, 267, 504, 505, 506, 507, 266, 267, 504, 505, 506, 507, 266, 267, 506, 507, 508, 266, 267, 506, 507, 509, 510, 515, 104, 270, 512, 102, 270, 511, 270, 271, 511, 512, 513, 514, 270, 271, 511, 512, 513, 514, 270, 271, 513, 514, 515, 270, 271, 513, 514, 516, 517, 522, 108, 274, 519, 106, 274, 518, 274, 275, 518, 519, 520, 521, 274, 275, 518, 519, 520, 521, 274, 275, 520, 521, 522, 274, 275, 520, 521, 523, 524, 529, 112, 278, 526, 110, 278, 525, 278, 279, 525, 526, 527, 528, 278, 279, 525, 526, 527, 528, 278, 279, 527, 528, 529, 278, 279, 527, 528, 530, 531, 536, 116, 282, 533, 114, 282, 532, 282, 283, 532, 533, 534, 535, 282, 283, 532, 533, 534, 535, 282, 283, 534, 535, 536, 282, 283, 534, 535, 537, 538, 543, 120, 286, 540, 118, 286, 539, 286, 287, 539, 540, 541, 542, 286, 287, 539, 540, 541, 542, 286, 287, 541, 542, 543, 286, 287, 541, 542, 544, 545, 550, 124, 290, 547, 122, 290, 546, 290, 291, 546, 547, 548, 549, 290, 291, 546, 547, 548, 549, 290, 291, 548, 549, 550, 290, 291, 548, 549, 551, 552, 557, 128, 294, 554, 126, 294, 553, 294, 295, 553, 554, 555, 556, 294, 295, 553, 554, 555, 556, 294, 295, 555, 556, 557, 294, 295, 555, 556, 558, 559, 564, 132, 298, 561, 130, 298, 560, 298, 299, 560, 561, 562, 563, 298, 299, 560, 561, 562, 563, 298, 299, 562, 563, 564, 298, 299, 562, 563, 565, 566, 571, 136, 302, 568, 134, 302, 567, 302, 303, 567, 568, 569, 570, 302, 303, 567, 568, 569, 570, 302, 303, 569, 570, 571, 302, 303, 569, 570, 572, 573, 578, 140, 306, 575, 138, 306, 574, 306, 307, 574, 575, 576, 577, 306, 307, 574, 575, 576, 577, 306, 307, 576, 577, 578, 306, 307, 576, 577, 579, 580, 585, 144, 310, 582, 142, 310, 581, 310, 311, 581, 582, 583, 584, 310, 311, 581, 582, 583, 584, 310, 311, 583, 584, 585, 310, 311, 583, 584, 586, 587, 592, 148, 314, 589, 146, 314, 588, 314, 315, 588, 589, 590, 591, 314, 315, 588, 589, 590, 591, 314, 315, 590, 591, 592, 314, 315, 590, 591, 593, 594, 599, 152, 318, 596, 150, 318, 595, 318, 319, 595, 596, 597, 598, 318, 319, 595, 596, 597, 598, 318, 319, 597, 598, 599, 318, 319, 597, 598, 600, 601, 606, 156, 322, 603, 154, 322, 602, 322, 323, 602, 603, 604, 605, 322, 323, 602, 603, 604, 605, 322, 323, 604, 605, 606, 322, 323, 604, 605, 607, 608, 613, 160, 326, 610, 158, 326, 609, 326, 327, 609, 610, 611, 612, 326, 327, 609, 610, 611, 612, 326, 327, 611, 612, 613, 326, 327, 611, 612, 614, 615, 620, 164, 330, 617, 162, 330, 616, 330, 331, 616, 617, 618, 619, 330, 331, 616, 617, 618, 619, 330, 331, 618, 619, 620, 330, 331, 618, 619, 621, 337, 622, 165, 622, 623]
    sp_jac_run_ja = [0, 2, 3, 6, 7, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29, 30, 32, 33, 35, 36, 38, 39, 41, 42, 44, 45, 47, 48, 50, 51, 53, 54, 56, 57, 59, 60, 62, 63, 65, 66, 68, 69, 71, 72, 74, 75, 77, 78, 80, 81, 83, 84, 86, 87, 89, 90, 92, 93, 95, 96, 98, 99, 101, 102, 104, 105, 107, 108, 110, 111, 113, 114, 116, 117, 119, 120, 122, 123, 125, 126, 128, 129, 131, 132, 134, 135, 137, 138, 140, 141, 143, 144, 146, 147, 149, 150, 152, 153, 155, 156, 158, 159, 161, 162, 164, 165, 167, 168, 170, 171, 173, 174, 176, 177, 179, 180, 182, 183, 185, 186, 188, 189, 191, 192, 194, 195, 197, 198, 200, 201, 203, 204, 206, 207, 209, 210, 212, 213, 215, 216, 218, 219, 221, 222, 224, 225, 227, 228, 230, 231, 233, 234, 236, 237, 239, 240, 242, 243, 245, 246, 248, 250, 260, 270, 276, 282, 287, 292, 298, 302, 307, 312, 320, 328, 333, 338, 346, 354, 359, 364, 372, 380, 385, 390, 398, 406, 411, 416, 424, 432, 437, 442, 450, 458, 463, 468, 476, 484, 489, 494, 502, 510, 515, 520, 528, 536, 541, 546, 554, 562, 567, 572, 580, 588, 593, 598, 606, 614, 619, 624, 632, 640, 645, 650, 658, 666, 671, 676, 684, 692, 697, 702, 710, 718, 723, 728, 736, 744, 749, 754, 762, 770, 775, 780, 788, 796, 801, 806, 812, 818, 823, 828, 836, 844, 849, 854, 862, 870, 875, 880, 888, 896, 901, 906, 914, 922, 927, 932, 940, 948, 953, 958, 966, 974, 979, 984, 992, 1000, 1005, 1010, 1018, 1026, 1031, 1036, 1044, 1052, 1057, 1062, 1070, 1078, 1083, 1088, 1096, 1104, 1109, 1114, 1122, 1130, 1135, 1140, 1148, 1156, 1161, 1166, 1174, 1182, 1187, 1192, 1200, 1208, 1213, 1218, 1226, 1234, 1239, 1244, 1252, 1260, 1265, 1270, 1278, 1286, 1291, 1296, 1304, 1312, 1317, 1322, 1328, 1334, 1338, 1341, 1344, 1346, 1351, 1357, 1363, 1369, 1371, 1374, 1377, 1383, 1389, 1394, 1399, 1401, 1404, 1407, 1413, 1419, 1424, 1429, 1431, 1434, 1437, 1443, 1449, 1454, 1459, 1461, 1464, 1467, 1473, 1479, 1484, 1489, 1491, 1494, 1497, 1503, 1509, 1514, 1519, 1521, 1524, 1527, 1533, 1539, 1544, 1549, 1551, 1554, 1557, 1563, 1569, 1574, 1579, 1581, 1584, 1587, 1593, 1599, 1604, 1609, 1611, 1614, 1617, 1623, 1629, 1634, 1639, 1641, 1644, 1647, 1653, 1659, 1664, 1669, 1671, 1674, 1677, 1683, 1689, 1694, 1699, 1701, 1704, 1707, 1713, 1719, 1724, 1729, 1731, 1734, 1737, 1743, 1749, 1754, 1759, 1761, 1764, 1767, 1773, 1779, 1784, 1789, 1791, 1794, 1797, 1803, 1809, 1814, 1819, 1821, 1824, 1827, 1833, 1839, 1844, 1849, 1851, 1854, 1857, 1863, 1869, 1874, 1879, 1881, 1884, 1887, 1893, 1899, 1904, 1909, 1911, 1914, 1917, 1923, 1929, 1934, 1939, 1941, 1944, 1947, 1953, 1959, 1964, 1969, 1971, 1974, 1977, 1983, 1989, 1994, 1999, 2001, 2004, 2007, 2013, 2019, 2024, 2029, 2031, 2034, 2037, 2043, 2049, 2054, 2059, 2061, 2064, 2067, 2073, 2079, 2084, 2089, 2091, 2094, 2097, 2103, 2109, 2114, 2119, 2121, 2124, 2127, 2133, 2139, 2144, 2149, 2151, 2154, 2157, 2163, 2169, 2174, 2179, 2181, 2184, 2187, 2193, 2199, 2204, 2209, 2211, 2214, 2217, 2223, 2229, 2234, 2239, 2241, 2244, 2247, 2253, 2259, 2264, 2269, 2271, 2274, 2277, 2283, 2289, 2294, 2299, 2301, 2304, 2307, 2313, 2319, 2324, 2329, 2331, 2334, 2337, 2343, 2349, 2354, 2359, 2361, 2364, 2367, 2373, 2379, 2384, 2389, 2391, 2394, 2397, 2403, 2409, 2414, 2419, 2421, 2424, 2427, 2433, 2439, 2444, 2449, 2451, 2454, 2457, 2463, 2469, 2474, 2479, 2481, 2484, 2487, 2493, 2499, 2504, 2509, 2511, 2514, 2517, 2523, 2529, 2534, 2539, 2541, 2544, 2547, 2553, 2559, 2564, 2569, 2571, 2574]
    sp_jac_run_nia = 624
    sp_jac_run_nja = 624
    return sp_jac_run_ia, sp_jac_run_ja, sp_jac_run_nia, sp_jac_run_nja 

def sp_jac_trap_vectors():

    sp_jac_trap_ia = [0, 335, 0, 1, 2, 337, 622, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 13, 14, 15, 15, 16, 17, 17, 18, 19, 19, 20, 21, 21, 22, 23, 23, 24, 25, 25, 26, 27, 27, 28, 29, 29, 30, 31, 31, 32, 33, 33, 34, 35, 35, 36, 37, 37, 38, 39, 39, 40, 41, 41, 42, 43, 43, 44, 45, 45, 46, 47, 47, 48, 49, 49, 50, 51, 51, 52, 53, 53, 54, 55, 55, 56, 57, 57, 58, 59, 59, 60, 61, 61, 62, 63, 63, 64, 65, 65, 66, 67, 67, 68, 69, 69, 70, 71, 71, 72, 73, 73, 74, 75, 75, 76, 77, 77, 78, 79, 79, 80, 81, 81, 82, 83, 83, 84, 85, 85, 86, 87, 87, 88, 89, 89, 90, 91, 91, 92, 93, 93, 94, 95, 95, 96, 97, 97, 98, 99, 99, 100, 101, 101, 102, 103, 103, 104, 105, 105, 106, 107, 107, 108, 109, 109, 110, 111, 111, 112, 113, 113, 114, 115, 115, 116, 117, 117, 118, 119, 119, 120, 121, 121, 122, 123, 123, 124, 125, 125, 126, 127, 127, 128, 129, 129, 130, 131, 131, 132, 133, 133, 134, 135, 135, 136, 137, 137, 138, 139, 139, 140, 141, 141, 142, 143, 143, 144, 145, 145, 146, 147, 147, 148, 149, 149, 150, 151, 151, 152, 153, 153, 154, 155, 155, 156, 157, 157, 158, 159, 159, 160, 161, 161, 162, 163, 163, 164, 165, 622, 166, 167, 168, 169, 172, 173, 176, 177, 256, 257, 166, 167, 168, 169, 172, 173, 176, 177, 256, 257, 166, 167, 168, 169, 170, 171, 166, 167, 168, 169, 170, 171, 168, 169, 170, 171, 340, 168, 169, 170, 171, 341, 0, 1, 166, 167, 172, 173, 166, 167, 172, 173, 174, 175, 176, 177, 347, 174, 175, 176, 177, 348, 166, 167, 174, 175, 176, 177, 180, 181, 166, 167, 174, 175, 176, 177, 180, 181, 178, 179, 180, 181, 354, 178, 179, 180, 181, 355, 176, 177, 178, 179, 180, 181, 184, 185, 176, 177, 178, 179, 180, 181, 184, 185, 182, 183, 184, 185, 361, 182, 183, 184, 185, 362, 180, 181, 182, 183, 184, 185, 188, 189, 180, 181, 182, 183, 184, 185, 188, 189, 186, 187, 188, 189, 368, 186, 187, 188, 189, 369, 184, 185, 186, 187, 188, 189, 192, 193, 184, 185, 186, 187, 188, 189, 192, 193, 190, 191, 192, 193, 375, 190, 191, 192, 193, 376, 188, 189, 190, 191, 192, 193, 196, 197, 188, 189, 190, 191, 192, 193, 196, 197, 194, 195, 196, 197, 382, 194, 195, 196, 197, 383, 192, 193, 194, 195, 196, 197, 200, 201, 192, 193, 194, 195, 196, 197, 200, 201, 198, 199, 200, 201, 389, 198, 199, 200, 201, 390, 196, 197, 198, 199, 200, 201, 204, 205, 196, 197, 198, 199, 200, 201, 204, 205, 202, 203, 204, 205, 396, 202, 203, 204, 205, 397, 200, 201, 202, 203, 204, 205, 208, 209, 200, 201, 202, 203, 204, 205, 208, 209, 206, 207, 208, 209, 403, 206, 207, 208, 209, 404, 204, 205, 206, 207, 208, 209, 212, 213, 204, 205, 206, 207, 208, 209, 212, 213, 210, 211, 212, 213, 410, 210, 211, 212, 213, 411, 208, 209, 210, 211, 212, 213, 216, 217, 208, 209, 210, 211, 212, 213, 216, 217, 214, 215, 216, 217, 417, 214, 215, 216, 217, 418, 212, 213, 214, 215, 216, 217, 220, 221, 212, 213, 214, 215, 216, 217, 220, 221, 218, 219, 220, 221, 424, 218, 219, 220, 221, 425, 216, 217, 218, 219, 220, 221, 224, 225, 216, 217, 218, 219, 220, 221, 224, 225, 222, 223, 224, 225, 431, 222, 223, 224, 225, 432, 220, 221, 222, 223, 224, 225, 228, 229, 220, 221, 222, 223, 224, 225, 228, 229, 226, 227, 228, 229, 438, 226, 227, 228, 229, 439, 224, 225, 226, 227, 228, 229, 232, 233, 224, 225, 226, 227, 228, 229, 232, 233, 230, 231, 232, 233, 445, 230, 231, 232, 233, 446, 228, 229, 230, 231, 232, 233, 236, 237, 228, 229, 230, 231, 232, 233, 236, 237, 234, 235, 236, 237, 452, 234, 235, 236, 237, 453, 232, 233, 234, 235, 236, 237, 240, 241, 232, 233, 234, 235, 236, 237, 240, 241, 238, 239, 240, 241, 459, 238, 239, 240, 241, 460, 236, 237, 238, 239, 240, 241, 244, 245, 236, 237, 238, 239, 240, 241, 244, 245, 242, 243, 244, 245, 466, 242, 243, 244, 245, 467, 240, 241, 242, 243, 244, 245, 248, 249, 240, 241, 242, 243, 244, 245, 248, 249, 246, 247, 248, 249, 473, 246, 247, 248, 249, 474, 244, 245, 246, 247, 248, 249, 252, 253, 244, 245, 246, 247, 248, 249, 252, 253, 250, 251, 252, 253, 480, 250, 251, 252, 253, 481, 248, 249, 250, 251, 252, 253, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 487, 254, 255, 256, 257, 488, 166, 167, 254, 255, 256, 257, 260, 261, 166, 167, 254, 255, 256, 257, 260, 261, 258, 259, 260, 261, 494, 258, 259, 260, 261, 495, 256, 257, 258, 259, 260, 261, 264, 265, 256, 257, 258, 259, 260, 261, 264, 265, 262, 263, 264, 265, 501, 262, 263, 264, 265, 502, 260, 261, 262, 263, 264, 265, 268, 269, 260, 261, 262, 263, 264, 265, 268, 269, 266, 267, 268, 269, 508, 266, 267, 268, 269, 509, 264, 265, 266, 267, 268, 269, 272, 273, 264, 265, 266, 267, 268, 269, 272, 273, 270, 271, 272, 273, 515, 270, 271, 272, 273, 516, 268, 269, 270, 271, 272, 273, 276, 277, 268, 269, 270, 271, 272, 273, 276, 277, 274, 275, 276, 277, 522, 274, 275, 276, 277, 523, 272, 273, 274, 275, 276, 277, 280, 281, 272, 273, 274, 275, 276, 277, 280, 281, 278, 279, 280, 281, 529, 278, 279, 280, 281, 530, 276, 277, 278, 279, 280, 281, 284, 285, 276, 277, 278, 279, 280, 281, 284, 285, 282, 283, 284, 285, 536, 282, 283, 284, 285, 537, 280, 281, 282, 283, 284, 285, 288, 289, 280, 281, 282, 283, 284, 285, 288, 289, 286, 287, 288, 289, 543, 286, 287, 288, 289, 544, 284, 285, 286, 287, 288, 289, 292, 293, 284, 285, 286, 287, 288, 289, 292, 293, 290, 291, 292, 293, 550, 290, 291, 292, 293, 551, 288, 289, 290, 291, 292, 293, 296, 297, 288, 289, 290, 291, 292, 293, 296, 297, 294, 295, 296, 297, 557, 294, 295, 296, 297, 558, 292, 293, 294, 295, 296, 297, 300, 301, 292, 293, 294, 295, 296, 297, 300, 301, 298, 299, 300, 301, 564, 298, 299, 300, 301, 565, 296, 297, 298, 299, 300, 301, 304, 305, 296, 297, 298, 299, 300, 301, 304, 305, 302, 303, 304, 305, 571, 302, 303, 304, 305, 572, 300, 301, 302, 303, 304, 305, 308, 309, 300, 301, 302, 303, 304, 305, 308, 309, 306, 307, 308, 309, 578, 306, 307, 308, 309, 579, 304, 305, 306, 307, 308, 309, 312, 313, 304, 305, 306, 307, 308, 309, 312, 313, 310, 311, 312, 313, 585, 310, 311, 312, 313, 586, 308, 309, 310, 311, 312, 313, 316, 317, 308, 309, 310, 311, 312, 313, 316, 317, 314, 315, 316, 317, 592, 314, 315, 316, 317, 593, 312, 313, 314, 315, 316, 317, 320, 321, 312, 313, 314, 315, 316, 317, 320, 321, 318, 319, 320, 321, 599, 318, 319, 320, 321, 600, 316, 317, 318, 319, 320, 321, 324, 325, 316, 317, 318, 319, 320, 321, 324, 325, 322, 323, 324, 325, 606, 322, 323, 324, 325, 607, 320, 321, 322, 323, 324, 325, 328, 329, 320, 321, 322, 323, 324, 325, 328, 329, 326, 327, 328, 329, 613, 326, 327, 328, 329, 614, 324, 325, 326, 327, 328, 329, 332, 333, 324, 325, 326, 327, 328, 329, 332, 333, 330, 331, 332, 333, 620, 330, 331, 332, 333, 621, 328, 329, 330, 331, 332, 333, 328, 329, 330, 331, 332, 333, 0, 1, 172, 334, 334, 335, 336, 0, 335, 336, 3, 337, 2, 170, 171, 338, 339, 2, 4, 170, 171, 338, 339, 2, 170, 171, 338, 339, 340, 2, 170, 171, 338, 339, 341, 342, 347, 8, 174, 344, 6, 174, 343, 174, 175, 343, 344, 345, 346, 174, 175, 343, 344, 345, 346, 174, 175, 345, 346, 347, 174, 175, 345, 346, 348, 349, 354, 12, 178, 351, 10, 178, 350, 178, 179, 350, 351, 352, 353, 178, 179, 350, 351, 352, 353, 178, 179, 352, 353, 354, 178, 179, 352, 353, 355, 356, 361, 16, 182, 358, 14, 182, 357, 182, 183, 357, 358, 359, 360, 182, 183, 357, 358, 359, 360, 182, 183, 359, 360, 361, 182, 183, 359, 360, 362, 363, 368, 20, 186, 365, 18, 186, 364, 186, 187, 364, 365, 366, 367, 186, 187, 364, 365, 366, 367, 186, 187, 366, 367, 368, 186, 187, 366, 367, 369, 370, 375, 24, 190, 372, 22, 190, 371, 190, 191, 371, 372, 373, 374, 190, 191, 371, 372, 373, 374, 190, 191, 373, 374, 375, 190, 191, 373, 374, 376, 377, 382, 28, 194, 379, 26, 194, 378, 194, 195, 378, 379, 380, 381, 194, 195, 378, 379, 380, 381, 194, 195, 380, 381, 382, 194, 195, 380, 381, 383, 384, 389, 32, 198, 386, 30, 198, 385, 198, 199, 385, 386, 387, 388, 198, 199, 385, 386, 387, 388, 198, 199, 387, 388, 389, 198, 199, 387, 388, 390, 391, 396, 36, 202, 393, 34, 202, 392, 202, 203, 392, 393, 394, 395, 202, 203, 392, 393, 394, 395, 202, 203, 394, 395, 396, 202, 203, 394, 395, 397, 398, 403, 40, 206, 400, 38, 206, 399, 206, 207, 399, 400, 401, 402, 206, 207, 399, 400, 401, 402, 206, 207, 401, 402, 403, 206, 207, 401, 402, 404, 405, 410, 44, 210, 407, 42, 210, 406, 210, 211, 406, 407, 408, 409, 210, 211, 406, 407, 408, 409, 210, 211, 408, 409, 410, 210, 211, 408, 409, 411, 412, 417, 48, 214, 414, 46, 214, 413, 214, 215, 413, 414, 415, 416, 214, 215, 413, 414, 415, 416, 214, 215, 415, 416, 417, 214, 215, 415, 416, 418, 419, 424, 52, 218, 421, 50, 218, 420, 218, 219, 420, 421, 422, 423, 218, 219, 420, 421, 422, 423, 218, 219, 422, 423, 424, 218, 219, 422, 423, 425, 426, 431, 56, 222, 428, 54, 222, 427, 222, 223, 427, 428, 429, 430, 222, 223, 427, 428, 429, 430, 222, 223, 429, 430, 431, 222, 223, 429, 430, 432, 433, 438, 60, 226, 435, 58, 226, 434, 226, 227, 434, 435, 436, 437, 226, 227, 434, 435, 436, 437, 226, 227, 436, 437, 438, 226, 227, 436, 437, 439, 440, 445, 64, 230, 442, 62, 230, 441, 230, 231, 441, 442, 443, 444, 230, 231, 441, 442, 443, 444, 230, 231, 443, 444, 445, 230, 231, 443, 444, 446, 447, 452, 68, 234, 449, 66, 234, 448, 234, 235, 448, 449, 450, 451, 234, 235, 448, 449, 450, 451, 234, 235, 450, 451, 452, 234, 235, 450, 451, 453, 454, 459, 72, 238, 456, 70, 238, 455, 238, 239, 455, 456, 457, 458, 238, 239, 455, 456, 457, 458, 238, 239, 457, 458, 459, 238, 239, 457, 458, 460, 461, 466, 76, 242, 463, 74, 242, 462, 242, 243, 462, 463, 464, 465, 242, 243, 462, 463, 464, 465, 242, 243, 464, 465, 466, 242, 243, 464, 465, 467, 468, 473, 80, 246, 470, 78, 246, 469, 246, 247, 469, 470, 471, 472, 246, 247, 469, 470, 471, 472, 246, 247, 471, 472, 473, 246, 247, 471, 472, 474, 475, 480, 84, 250, 477, 82, 250, 476, 250, 251, 476, 477, 478, 479, 250, 251, 476, 477, 478, 479, 250, 251, 478, 479, 480, 250, 251, 478, 479, 481, 482, 487, 88, 254, 484, 86, 254, 483, 254, 255, 483, 484, 485, 486, 254, 255, 483, 484, 485, 486, 254, 255, 485, 486, 487, 254, 255, 485, 486, 488, 489, 494, 92, 258, 491, 90, 258, 490, 258, 259, 490, 491, 492, 493, 258, 259, 490, 491, 492, 493, 258, 259, 492, 493, 494, 258, 259, 492, 493, 495, 496, 501, 96, 262, 498, 94, 262, 497, 262, 263, 497, 498, 499, 500, 262, 263, 497, 498, 499, 500, 262, 263, 499, 500, 501, 262, 263, 499, 500, 502, 503, 508, 100, 266, 505, 98, 266, 504, 266, 267, 504, 505, 506, 507, 266, 267, 504, 505, 506, 507, 266, 267, 506, 507, 508, 266, 267, 506, 507, 509, 510, 515, 104, 270, 512, 102, 270, 511, 270, 271, 511, 512, 513, 514, 270, 271, 511, 512, 513, 514, 270, 271, 513, 514, 515, 270, 271, 513, 514, 516, 517, 522, 108, 274, 519, 106, 274, 518, 274, 275, 518, 519, 520, 521, 274, 275, 518, 519, 520, 521, 274, 275, 520, 521, 522, 274, 275, 520, 521, 523, 524, 529, 112, 278, 526, 110, 278, 525, 278, 279, 525, 526, 527, 528, 278, 279, 525, 526, 527, 528, 278, 279, 527, 528, 529, 278, 279, 527, 528, 530, 531, 536, 116, 282, 533, 114, 282, 532, 282, 283, 532, 533, 534, 535, 282, 283, 532, 533, 534, 535, 282, 283, 534, 535, 536, 282, 283, 534, 535, 537, 538, 543, 120, 286, 540, 118, 286, 539, 286, 287, 539, 540, 541, 542, 286, 287, 539, 540, 541, 542, 286, 287, 541, 542, 543, 286, 287, 541, 542, 544, 545, 550, 124, 290, 547, 122, 290, 546, 290, 291, 546, 547, 548, 549, 290, 291, 546, 547, 548, 549, 290, 291, 548, 549, 550, 290, 291, 548, 549, 551, 552, 557, 128, 294, 554, 126, 294, 553, 294, 295, 553, 554, 555, 556, 294, 295, 553, 554, 555, 556, 294, 295, 555, 556, 557, 294, 295, 555, 556, 558, 559, 564, 132, 298, 561, 130, 298, 560, 298, 299, 560, 561, 562, 563, 298, 299, 560, 561, 562, 563, 298, 299, 562, 563, 564, 298, 299, 562, 563, 565, 566, 571, 136, 302, 568, 134, 302, 567, 302, 303, 567, 568, 569, 570, 302, 303, 567, 568, 569, 570, 302, 303, 569, 570, 571, 302, 303, 569, 570, 572, 573, 578, 140, 306, 575, 138, 306, 574, 306, 307, 574, 575, 576, 577, 306, 307, 574, 575, 576, 577, 306, 307, 576, 577, 578, 306, 307, 576, 577, 579, 580, 585, 144, 310, 582, 142, 310, 581, 310, 311, 581, 582, 583, 584, 310, 311, 581, 582, 583, 584, 310, 311, 583, 584, 585, 310, 311, 583, 584, 586, 587, 592, 148, 314, 589, 146, 314, 588, 314, 315, 588, 589, 590, 591, 314, 315, 588, 589, 590, 591, 314, 315, 590, 591, 592, 314, 315, 590, 591, 593, 594, 599, 152, 318, 596, 150, 318, 595, 318, 319, 595, 596, 597, 598, 318, 319, 595, 596, 597, 598, 318, 319, 597, 598, 599, 318, 319, 597, 598, 600, 601, 606, 156, 322, 603, 154, 322, 602, 322, 323, 602, 603, 604, 605, 322, 323, 602, 603, 604, 605, 322, 323, 604, 605, 606, 322, 323, 604, 605, 607, 608, 613, 160, 326, 610, 158, 326, 609, 326, 327, 609, 610, 611, 612, 326, 327, 609, 610, 611, 612, 326, 327, 611, 612, 613, 326, 327, 611, 612, 614, 615, 620, 164, 330, 617, 162, 330, 616, 330, 331, 616, 617, 618, 619, 330, 331, 616, 617, 618, 619, 330, 331, 618, 619, 620, 330, 331, 618, 619, 621, 337, 622, 165, 622, 623]
    sp_jac_trap_ja = [0, 2, 4, 7, 8, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33, 34, 36, 37, 39, 40, 42, 43, 45, 46, 48, 49, 51, 52, 54, 55, 57, 58, 60, 61, 63, 64, 66, 67, 69, 70, 72, 73, 75, 76, 78, 79, 81, 82, 84, 85, 87, 88, 90, 91, 93, 94, 96, 97, 99, 100, 102, 103, 105, 106, 108, 109, 111, 112, 114, 115, 117, 118, 120, 121, 123, 124, 126, 127, 129, 130, 132, 133, 135, 136, 138, 139, 141, 142, 144, 145, 147, 148, 150, 151, 153, 154, 156, 157, 159, 160, 162, 163, 165, 166, 168, 169, 171, 172, 174, 175, 177, 178, 180, 181, 183, 184, 186, 187, 189, 190, 192, 193, 195, 196, 198, 199, 201, 202, 204, 205, 207, 208, 210, 211, 213, 214, 216, 217, 219, 220, 222, 223, 225, 226, 228, 229, 231, 232, 234, 235, 237, 238, 240, 241, 243, 244, 246, 247, 249, 251, 261, 271, 277, 283, 288, 293, 299, 303, 308, 313, 321, 329, 334, 339, 347, 355, 360, 365, 373, 381, 386, 391, 399, 407, 412, 417, 425, 433, 438, 443, 451, 459, 464, 469, 477, 485, 490, 495, 503, 511, 516, 521, 529, 537, 542, 547, 555, 563, 568, 573, 581, 589, 594, 599, 607, 615, 620, 625, 633, 641, 646, 651, 659, 667, 672, 677, 685, 693, 698, 703, 711, 719, 724, 729, 737, 745, 750, 755, 763, 771, 776, 781, 789, 797, 802, 807, 813, 819, 824, 829, 837, 845, 850, 855, 863, 871, 876, 881, 889, 897, 902, 907, 915, 923, 928, 933, 941, 949, 954, 959, 967, 975, 980, 985, 993, 1001, 1006, 1011, 1019, 1027, 1032, 1037, 1045, 1053, 1058, 1063, 1071, 1079, 1084, 1089, 1097, 1105, 1110, 1115, 1123, 1131, 1136, 1141, 1149, 1157, 1162, 1167, 1175, 1183, 1188, 1193, 1201, 1209, 1214, 1219, 1227, 1235, 1240, 1245, 1253, 1261, 1266, 1271, 1279, 1287, 1292, 1297, 1305, 1313, 1318, 1323, 1329, 1335, 1339, 1342, 1345, 1347, 1352, 1358, 1364, 1370, 1372, 1375, 1378, 1384, 1390, 1395, 1400, 1402, 1405, 1408, 1414, 1420, 1425, 1430, 1432, 1435, 1438, 1444, 1450, 1455, 1460, 1462, 1465, 1468, 1474, 1480, 1485, 1490, 1492, 1495, 1498, 1504, 1510, 1515, 1520, 1522, 1525, 1528, 1534, 1540, 1545, 1550, 1552, 1555, 1558, 1564, 1570, 1575, 1580, 1582, 1585, 1588, 1594, 1600, 1605, 1610, 1612, 1615, 1618, 1624, 1630, 1635, 1640, 1642, 1645, 1648, 1654, 1660, 1665, 1670, 1672, 1675, 1678, 1684, 1690, 1695, 1700, 1702, 1705, 1708, 1714, 1720, 1725, 1730, 1732, 1735, 1738, 1744, 1750, 1755, 1760, 1762, 1765, 1768, 1774, 1780, 1785, 1790, 1792, 1795, 1798, 1804, 1810, 1815, 1820, 1822, 1825, 1828, 1834, 1840, 1845, 1850, 1852, 1855, 1858, 1864, 1870, 1875, 1880, 1882, 1885, 1888, 1894, 1900, 1905, 1910, 1912, 1915, 1918, 1924, 1930, 1935, 1940, 1942, 1945, 1948, 1954, 1960, 1965, 1970, 1972, 1975, 1978, 1984, 1990, 1995, 2000, 2002, 2005, 2008, 2014, 2020, 2025, 2030, 2032, 2035, 2038, 2044, 2050, 2055, 2060, 2062, 2065, 2068, 2074, 2080, 2085, 2090, 2092, 2095, 2098, 2104, 2110, 2115, 2120, 2122, 2125, 2128, 2134, 2140, 2145, 2150, 2152, 2155, 2158, 2164, 2170, 2175, 2180, 2182, 2185, 2188, 2194, 2200, 2205, 2210, 2212, 2215, 2218, 2224, 2230, 2235, 2240, 2242, 2245, 2248, 2254, 2260, 2265, 2270, 2272, 2275, 2278, 2284, 2290, 2295, 2300, 2302, 2305, 2308, 2314, 2320, 2325, 2330, 2332, 2335, 2338, 2344, 2350, 2355, 2360, 2362, 2365, 2368, 2374, 2380, 2385, 2390, 2392, 2395, 2398, 2404, 2410, 2415, 2420, 2422, 2425, 2428, 2434, 2440, 2445, 2450, 2452, 2455, 2458, 2464, 2470, 2475, 2480, 2482, 2485, 2488, 2494, 2500, 2505, 2510, 2512, 2515, 2518, 2524, 2530, 2535, 2540, 2542, 2545, 2548, 2554, 2560, 2565, 2570, 2572, 2575]
    sp_jac_trap_nia = 624
    sp_jac_trap_nja = 624
    return sp_jac_trap_ia, sp_jac_trap_ja, sp_jac_trap_nia, sp_jac_trap_nja 
