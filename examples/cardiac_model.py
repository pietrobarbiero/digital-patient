# Size of variable arrays:
sizeAlgebraic = 51
sizeStates = 22
sizeConstants = 57
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (ms)"
    legend_states[0] = "V in component cell (mV)"
    legend_constants[0] = "R in component cell (joule_per_mole_kelvin)"
    legend_constants[1] = "T in component cell (kelvin)"
    legend_constants[2] = "F in component cell (coulomb_per_millimole)"
    legend_constants[43] = "FonRT in component cell (per_mV)"
    legend_constants[3] = "Cm in component cell (pF)"
    legend_constants[4] = "failing in component cell (dimensionless)"
    legend_algebraic[21] = "i_Na in component INa (uA_per_uF)"
    legend_algebraic[24] = "i_Ca in component ICa (uA_per_uF)"
    legend_algebraic[26] = "i_to in component Ito (uA_per_uF)"
    legend_algebraic[31] = "i_Kr in component IKr (uA_per_uF)"
    legend_algebraic[28] = "i_Ks in component IKs (uA_per_uF)"
    legend_algebraic[36] = "i_K1 in component IK1 (uA_per_uF)"
    legend_algebraic[41] = "i_NaCa in component INaCa (uA_per_uF)"
    legend_algebraic[40] = "i_NaK in component INaK (uA_per_uF)"
    legend_algebraic[38] = "i_b_Na in component INab (uA_per_uF)"
    legend_algebraic[37] = "i_b_Ca in component ICab (uA_per_uF)"
    legend_algebraic[44] = "dVdt in component cell (mV_per_ms)"
    legend_algebraic[42] = "I_tot in component cell (uA_per_uF)"
    legend_algebraic[10] = "i_Stim in component cell (uA_per_uF)"
    legend_constants[5] = "stim_offset in component cell (ms)"
    legend_constants[6] = "stim_period in component cell (ms)"
    legend_constants[7] = "stim_duration in component cell (ms)"
    legend_constants[8] = "stim_amplitude in component cell (uA_per_uF)"
    legend_algebraic[0] = "past in component cell (ms)"
    legend_constants[9] = "V_clamp in component cell (dimensionless)"
    legend_constants[10] = "V_hold in component cell (mV)"
    legend_constants[11] = "V_step in component cell (mV)"
    legend_algebraic[20] = "E_Na in component INa (mV)"
    legend_constants[12] = "g_Na in component INa (mS_per_uF)"
    legend_states[1] = "Nai in component Ionic_concentrations (mM)"
    legend_constants[13] = "Nao in component Ionic_concentrations (mM)"
    legend_states[2] = "m in component INa_m_gate (dimensionless)"
    legend_states[3] = "h in component INa_h_gate (dimensionless)"
    legend_states[4] = "j in component INa_j_gate (dimensionless)"
    legend_algebraic[1] = "alpha_m in component INa_m_gate (per_ms)"
    legend_algebraic[11] = "beta_m in component INa_m_gate (per_ms)"
    legend_algebraic[2] = "alpha_h in component INa_h_gate (per_ms)"
    legend_algebraic[12] = "beta_h in component INa_h_gate (per_ms)"
    legend_algebraic[3] = "alpha_j in component INa_j_gate (per_ms)"
    legend_algebraic[13] = "beta_j in component INa_j_gate (per_ms)"
    legend_algebraic[22] = "E_Ca in component ICa (mV)"
    legend_constants[14] = "g_Ca_max in component ICa (mS_per_uF)"
    legend_states[5] = "Cai in component Ionic_concentrations (mM)"
    legend_constants[15] = "Cao in component Ionic_concentrations (mM)"
    legend_states[6] = "d in component ICa_d_gate (dimensionless)"
    legend_states[7] = "f in component ICa_f_gate (dimensionless)"
    legend_algebraic[23] = "f_Ca in component ICa_f_Ca_gate (dimensionless)"
    legend_algebraic[4] = "alpha_d in component ICa_d_gate (per_ms)"
    legend_algebraic[14] = "beta_d in component ICa_d_gate (per_ms)"
    legend_algebraic[5] = "alpha_f in component ICa_f_gate (per_ms)"
    legend_algebraic[15] = "beta_f in component ICa_f_gate (per_ms)"
    legend_constants[16] = "Km_Ca in component ICa_f_Ca_gate (mM)"
    legend_constants[44] = "g_to_max in component Ito (mS_per_uF)"
    legend_algebraic[25] = "E_to in component Ito (mV)"
    legend_states[8] = "Ki in component Ionic_concentrations (mM)"
    legend_constants[17] = "Ko in component Ionic_concentrations (mM)"
    legend_states[9] = "t in component Ito_t_gate (dimensionless)"
    legend_states[10] = "r in component Ito_r_gate (dimensionless)"
    legend_algebraic[6] = "alpha_r in component Ito_r_gate (per_ms)"
    legend_algebraic[16] = "beta_r in component Ito_r_gate (per_ms)"
    legend_algebraic[7] = "alpha_t in component Ito_t_gate (per_ms)"
    legend_algebraic[17] = "beta_t in component Ito_t_gate (per_ms)"
    legend_constants[18] = "g_Ks_max in component IKs (mS_per_uF)"
    legend_algebraic[27] = "E_Ks in component IKs (mV)"
    legend_states[11] = "Xs in component IKs_Xs_gate (dimensionless)"
    legend_algebraic[8] = "alpha_Xs in component IKs_Xs_gate (per_ms)"
    legend_algebraic[18] = "beta_Xs in component IKs_Xs_gate (per_ms)"
    legend_constants[19] = "g_Kr_max in component IKr (mS_per_uF)"
    legend_algebraic[30] = "rik in component IKr (dimensionless)"
    legend_algebraic[29] = "E_K in component IKr (mV)"
    legend_states[12] = "Xr in component IKr_Xr_gate (dimensionless)"
    legend_algebraic[9] = "alpha_Xr in component IKr_Xr_gate (per_ms)"
    legend_algebraic[19] = "beta_Xr in component IKr_Xr_gate (per_ms)"
    legend_algebraic[32] = "E_K1 in component IK1 (mV)"
    legend_constants[45] = "g_K1_max in component IK1 (mS_per_uF)"
    legend_algebraic[35] = "K1_infinity in component IK1_K1_gate (dimensionless)"
    legend_algebraic[33] = "alpha_K1 in component IK1_K1_gate (per_ms)"
    legend_algebraic[34] = "beta_K1 in component IK1_K1_gate (per_ms)"
    legend_constants[46] = "g_b_Ca_max in component ICab (mS_per_uF)"
    legend_constants[47] = "g_b_Na_max in component INab (mS_per_uF)"
    legend_constants[48] = "I_NaK_max in component INaK (uA_per_uF)"
    legend_algebraic[39] = "f_NaK in component INaK (dimensionless)"
    legend_constants[20] = "K_mNai in component INaK (mM)"
    legend_constants[21] = "K_mKo in component INaK (mM)"
    legend_constants[49] = "sigma in component INaK (dimensionless)"
    legend_constants[50] = "K_NaCa in component INaCa (uA_per_uF)"
    legend_constants[22] = "K_mNa in component INaCa (mM)"
    legend_constants[23] = "K_mCa in component INaCa (mM)"
    legend_constants[24] = "K_sat in component INaCa (dimensionless)"
    legend_constants[25] = "eta in component INaCa (dimensionless)"
    legend_algebraic[45] = "i_rel in component Irel (mM_per_ms)"
    legend_algebraic[43] = "G_rel in component Irel (per_ms)"
    legend_constants[26] = "G_rel_max in component Irel (per_ms)"
    legend_constants[27] = "G_rel_overload in component Irel (per_ms)"
    legend_constants[28] = "K_mrel in component Irel (mM)"
    legend_constants[29] = "delta_Ca_ith in component Irel (mM)"
    legend_constants[30] = "K_mCSQN in component calcium_buffers_in_the_JSR (mM)"
    legend_states[13] = "Ca_JSR in component Ionic_concentrations (mM)"
    legend_constants[54] = "V_myo in component Ionic_concentrations (fL)"
    legend_constants[56] = "V_JSR in component Ionic_concentrations (fL)"
    legend_states[14] = "APtrack in component Irel (dimensionless)"
    legend_states[15] = "APtrack2 in component Irel (dimensionless)"
    legend_states[16] = "APtrack3 in component Irel (dimensionless)"
    legend_states[17] = "Cainfluxtrack in component Irel (mM)"
    legend_states[18] = "OVRLDtrack in component Irel (dimensionless)"
    legend_states[19] = "OVRLDtrack2 in component Irel (dimensionless)"
    legend_states[20] = "OVRLDtrack3 in component Irel (dimensionless)"
    legend_constants[31] = "CSQNthresh in component Irel (mM)"
    legend_constants[32] = "Logicthresh in component Irel (dimensionless)"
    legend_algebraic[46] = "i_up in component Iup (mM_per_ms)"
    legend_constants[51] = "I_up_max in component Iup (mM_per_ms)"
    legend_constants[33] = "K_mup in component Iup (mM)"
    legend_algebraic[47] = "i_leak in component Ileak (mM_per_ms)"
    legend_constants[52] = "K_leak in component Ileak (per_ms)"
    legend_states[21] = "Ca_NSR in component Ionic_concentrations (mM)"
    legend_algebraic[49] = "i_tr in component Itr (mM_per_ms)"
    legend_constants[34] = "tau_tr in component Itr (ms)"
    legend_constants[35] = "K_mTn in component calcium_buffers_in_the_myoplasm (mM)"
    legend_constants[36] = "K_mCMDN in component calcium_buffers_in_the_myoplasm (mM)"
    legend_constants[37] = "Tn_max in component calcium_buffers_in_the_myoplasm (mM)"
    legend_constants[38] = "CMDN_max in component calcium_buffers_in_the_myoplasm (mM)"
    legend_constants[39] = "buffon in component calcium_buffers_in_the_myoplasm (dimensionless)"
    legend_algebraic[48] = "Cai_bufc in component calcium_buffers_in_the_myoplasm (dimensionless)"
    legend_constants[40] = "CSQN_max in component calcium_buffers_in_the_JSR (mM)"
    legend_algebraic[50] = "Ca_JSR_bufc in component calcium_buffers_in_the_JSR (dimensionless)"
    legend_constants[41] = "preplength in component Ionic_concentrations (um)"
    legend_constants[42] = "radius in component Ionic_concentrations (um)"
    legend_constants[53] = "volume in component Ionic_concentrations (fL)"
    legend_constants[55] = "V_NSR in component Ionic_concentrations (fL)"
    legend_rates[0] = "d/dt V in component cell (mV)"
    legend_rates[2] = "d/dt m in component INa_m_gate (dimensionless)"
    legend_rates[3] = "d/dt h in component INa_h_gate (dimensionless)"
    legend_rates[4] = "d/dt j in component INa_j_gate (dimensionless)"
    legend_rates[6] = "d/dt d in component ICa_d_gate (dimensionless)"
    legend_rates[7] = "d/dt f in component ICa_f_gate (dimensionless)"
    legend_rates[10] = "d/dt r in component Ito_r_gate (dimensionless)"
    legend_rates[9] = "d/dt t in component Ito_t_gate (dimensionless)"
    legend_rates[11] = "d/dt Xs in component IKs_Xs_gate (dimensionless)"
    legend_rates[12] = "d/dt Xr in component IKr_Xr_gate (dimensionless)"
    legend_rates[14] = "d/dt APtrack in component Irel (dimensionless)"
    legend_rates[15] = "d/dt APtrack2 in component Irel (dimensionless)"
    legend_rates[16] = "d/dt APtrack3 in component Irel (dimensionless)"
    legend_rates[17] = "d/dt Cainfluxtrack in component Irel (mM)"
    legend_rates[18] = "d/dt OVRLDtrack in component Irel (dimensionless)"
    legend_rates[19] = "d/dt OVRLDtrack2 in component Irel (dimensionless)"
    legend_rates[20] = "d/dt OVRLDtrack3 in component Irel (dimensionless)"
    legend_rates[1] = "d/dt Nai in component Ionic_concentrations (mM)"
    legend_rates[8] = "d/dt Ki in component Ionic_concentrations (mM)"
    legend_rates[5] = "d/dt Cai in component Ionic_concentrations (mM)"
    legend_rates[13] = "d/dt Ca_JSR in component Ionic_concentrations (mM)"
    legend_rates[21] = "d/dt Ca_NSR in component Ionic_concentrations (mM)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = -90.7796417483135
    constants[0] = 8.3143
    constants[1] = 310.15
    constants[2] = 96.4867
    constants[3] = 153.4
    constants[4] = 0
    constants[5] = 0
    constants[6] = 1000
    constants[7] = 3
    constants[8] = -15
    constants[9] = 0
    constants[10] = -60
    constants[11] = 0
    constants[12] = 16
    states[1] = 10
    constants[13] = 138
    states[2] = 0.000585525582501575
    states[3] = 0.995865529216237
    states[4] = 0.997011204496203
    constants[14] = 0.064
    states[5] = 0.0002
    constants[15] = 2
    states[6] = 2.50653215966786e-10
    states[7] = 0.92130376850548
    constants[16] = 0.0006
    states[8] = 140
    constants[17] = 4
    states[9] = 0.999897251531651
    states[10] = 1.75032478501027e-5
    constants[18] = 0.02
    states[11] = 0.00885658064818147
    constants[19] = 0.015
    states[12] = 0.000215523048438941
    constants[20] = 10
    constants[21] = 1.5
    constants[22] = 87.5
    constants[23] = 1.38
    constants[24] = 0.1
    constants[25] = 0.35
    constants[26] = 22
    constants[27] = 3
    constants[28] = 0.0008
    constants[29] = 5e-6
    constants[30] = 0.8
    states[13] = 2.5
    states[14] = -1.372158997089e-136
    states[15] = -7.58517896402761e-136
    states[16] = 4.82035353592764e-5
    states[17] = -7.71120176147331e-138
    states[18] = 1e-6
    states[19] = 1e-6
    states[20] = 1e-6
    constants[31] = 0.7
    constants[32] = 0.98
    constants[33] = 0.00092
    states[21] = 2.5
    constants[34] = 180
    constants[35] = 0.0005
    constants[36] = 0.00238
    constants[37] = 0.07
    constants[38] = 0.05
    constants[39] = 1
    constants[40] = 10
    constants[41] = 100
    constants[42] = 11
    constants[43] = constants[2]/(constants[0]*constants[1])
    constants[44] = custom_piecewise([equal(constants[4] , 0.00000), 0.300000 , True, 0.191000])
    constants[45] = custom_piecewise([equal(constants[4] , 0.00000), 2.50000 , True, 2.00000])
    constants[46] = custom_piecewise([equal(constants[4] , 0.00000), 0.000850000 , True, 0.00130000])
    constants[47] = custom_piecewise([equal(constants[4] , 0.00000), 0.00100000 , True, 0.00000])
    constants[48] = custom_piecewise([equal(constants[4] , 0.00000), 1.30000 , True, 0.750000])
    constants[49] = (1.00000/7.00000)*(exp(constants[13]/67.3000)-1.00000)
    constants[50] = custom_piecewise([equal(constants[4] , 0.00000), 1000.00 , True, 1650.00])
    constants[51] = custom_piecewise([equal(constants[4] , 0.00000), 0.00450000 , True, 0.00150000])
    constants[52] = custom_piecewise([equal(constants[4] , 0.00000), 0.000260000 , True, 0.000170000])
    constants[53] =  pi*constants[41]*(power(constants[42], 2.00000))
    constants[54] = 0.680000*constants[53]
    constants[55] = 0.0552000*constants[53]
    constants[56] = 0.00480000*constants[53]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[15] = custom_piecewise([less(states[14] , 0.200000) & greater(states[14] , 0.180000), 100.000*(1.00000-states[15])-0.500000*states[15] , True, -0.500000*states[15]])
    rates[16] = custom_piecewise([less(states[14] , 0.200000) & greater(states[14] , 0.180000), 100.000*(1.00000-states[16])-0.500000*states[16] , True, -0.0100000*states[16]])
    rates[18] = custom_piecewise([greater(1.00000/(1.00000+constants[30]/states[13]) , constants[31]) & less(states[20] , 0.370000) & less(states[16] , 0.370000), 0.00000*50.0000*(1.00000-states[18]) , True, -0.00000*0.500000*states[18]])
    rates[19] = custom_piecewise([greater(states[18] , constants[32]) & less(states[19] , constants[32]), 0.00000*50.0000*(1.00000-states[19]) , True, -0.00000*0.500000*states[19]])
    rates[20] = custom_piecewise([greater(states[18] , constants[32]) & less(states[20] , constants[32]), 0.00000*50.0000*(1.00000-states[20]) , True, -0.00000*0.0100000*states[20]])
    algebraic[1] = custom_piecewise([greater(fabs(states[0]+47.1300) , 0.00100000), (0.320000*(states[0]+47.1300))/(1.00000-exp(-0.100000*(states[0]+47.1300))) , True, 3.20000])
    algebraic[11] = 0.0800000*exp(-states[0]/11.0000)
    rates[2] = algebraic[1]*(1.00000-states[2])-algebraic[11]*states[2]
    algebraic[2] = custom_piecewise([less(states[0] , -40.0000), 0.135000*exp((80.0000+states[0])/-6.80000) , True, 0.00000])
    algebraic[12] = custom_piecewise([less(states[0] , -40.0000), 3.56000*exp(0.0790000*states[0])+310000.*exp(0.350000*states[0]) , True, 1.00000/(0.130000*(1.00000+exp(-(states[0]+10.6600)/11.1000)))])
    rates[3] = algebraic[2]*(1.00000-states[3])-algebraic[12]*states[3]
    algebraic[3] = custom_piecewise([less(states[0] , -40.0000), ((-127140.*exp(0.244000*states[0])-3.47400e-05*exp(-0.0439100*states[0]))*(states[0]+37.7800))/(1.00000+exp(0.311000*(states[0]+79.2300))) , True, 0.00000])
    algebraic[13] = custom_piecewise([less(states[0] , -40.0000), (0.121200*exp(-0.0105200*states[0]))/(1.00000+exp(-0.137800*(states[0]+40.1400))) , True, (0.300000*exp(-2.53500e-07*states[0]))/(1.00000+exp(-0.100000*(states[0]+32.0000)))])
    rates[4] = algebraic[3]*(1.00000-states[4])-algebraic[13]*states[4]
    algebraic[4] = (14.9859/(16.6813*(power(2.00000* pi, 1.0/2))))*exp(-(power((states[0]-22.3600)/16.6813, 2.00000))/2.00000)
    algebraic[14] = 0.147100-(5.30000/(14.9300*(power(2.00000* pi, 1.0/2))))*exp(-(power((states[0]-6.27440)/14.9300, 2.00000))/2.00000)
    rates[6] = algebraic[4]*(1.00000-states[6])-algebraic[14]*states[6]
    algebraic[5] = 0.00687200/(1.00000+exp((states[0]-6.15460)/6.12230))
    algebraic[15] = (0.0687000*exp(-0.108100*(states[0]+9.82550))+0.0112000)/(1.00000+exp(-0.277900*(states[0]+9.82550)))+0.000547400
    rates[7] = algebraic[5]*(1.00000-states[7])-algebraic[15]*states[7]
    algebraic[6] = (0.526600*exp(-0.0166000*(states[0]-42.2912)))/(1.00000+exp(-0.0943000*(states[0]-42.2912)))
    algebraic[16] = (5.18600e-05*states[0]+0.514900*exp(-0.134400*(states[0]-5.00270)))/(1.00000+exp(-0.134800*(states[0]-5.18600e-05)))
    rates[10] = algebraic[6]*(1.00000-states[10])-algebraic[16]*states[10]
    algebraic[7] = (5.61200e-05*states[0]+0.0721000*exp(-0.173000*(states[0]+34.2531)))/(1.00000+exp(-0.173200*(states[0]+34.2531)))
    algebraic[17] = (0.000121500*states[0]+0.0767000*exp(-1.66000e-09*(states[0]+34.0235)))/(1.00000+exp(-0.160400*(states[0]+34.0235)))
    rates[9] = algebraic[7]*(1.00000-states[9])-algebraic[17]*states[9]
    algebraic[8] = 0.00301300/(1.00000+exp((7.44540-(states[0]+10.0000))/14.3171))
    algebraic[18] = 0.00587000/(1.00000+exp((5.95000+states[0]+10.0000)/15.8200))
    rates[11] = algebraic[8]*(1.00000-states[11])-algebraic[18]*states[11]
    algebraic[9] = (0.00500000*exp(0.000526600*(states[0]+4.06700)))/(1.00000+exp(-0.126200*(states[0]+4.06700)))
    algebraic[19] = (0.0160000*exp(0.00160000*(states[0]+65.6600)))/(1.00000+exp(0.0783000*(states[0]+65.6600)))
    rates[12] = algebraic[9]*(1.00000-states[12])-algebraic[19]*states[12]
    algebraic[25] = log((0.0430000*constants[13]+constants[17])/(0.0430000*states[1]+states[8]))/constants[43]
    algebraic[26] = constants[44]*states[10]*states[9]*(states[0]-algebraic[25])
    algebraic[30] = 1.00000/(1.00000+exp((states[0]+26.0000)/23.0000))
    algebraic[29] = log(constants[17]/states[8])/constants[43]
    algebraic[31] = constants[19]*states[12]*algebraic[30]*(states[0]-algebraic[29])
    algebraic[27] = log((0.0183300*constants[13]+constants[17])/(0.0183300*states[1]+states[8]))/constants[43]
    algebraic[28] = constants[18]*(power(states[11], 2.00000))*(states[0]-algebraic[27])
    algebraic[32] = log(constants[17]/states[8])/constants[43]
    algebraic[33] = 0.100000/(1.00000+exp(0.0600000*(states[0]-(algebraic[32]+200.000))))
    algebraic[34] = (3.00000*exp(0.000200000*(states[0]+100.000+-algebraic[32]))+1.00000*exp(0.100000*(states[0]-(10.0000+algebraic[32]))))/(1.00000+exp(-0.500000*(states[0]-algebraic[32])))
    algebraic[35] = algebraic[33]/(algebraic[33]+algebraic[34])
    algebraic[36] = constants[45]*algebraic[35]*(states[0]-algebraic[32])
    algebraic[39] = 1.00000/(1.00000+0.124500*exp(-0.100000*states[0]*constants[43])+0.0365000*constants[49]*exp(-states[0]*constants[43]))
    algebraic[40] = (((constants[48]*algebraic[39]*1.00000)/(1.00000+power(constants[20]/states[1], 1.50000)))*constants[17])/(constants[17]+constants[21])
    algebraic[0] = floor(voi/constants[6])*constants[6]
    algebraic[10] = custom_piecewise([greater_equal(voi-algebraic[0] , constants[5]) & less_equal(voi-algebraic[0] , constants[5]+constants[7]), constants[8] , True, 0.00000])
    rates[8] = (-1.00000*constants[3]*((algebraic[26]+algebraic[31]+algebraic[36]+algebraic[10]+algebraic[28])-2.00000*algebraic[40]))/(constants[54]*constants[2])
    algebraic[22] = log(constants[15]/states[5])/(2.00000*constants[43])
    algebraic[23] = constants[16]/(constants[16]+states[5])
    algebraic[24] = constants[14]*states[6]*states[7]*algebraic[23]*(states[0]-algebraic[22])
    algebraic[41] = ((((((constants[50]*1.00000)/(power(constants[22], 3.00000)+power(constants[13], 3.00000)))*1.00000)/(constants[23]+constants[15]))*1.00000)/(1.00000+constants[24]*exp((constants[25]-1.00000)*states[0]*constants[43])))*(exp(constants[25]*states[0]*constants[43])*(power(states[1], 3.00000))*constants[15]-exp((constants[25]-1.00000)*states[0]*constants[43])*(power(constants[13], 3.00000))*states[5])
    algebraic[37] = constants[46]*(states[0]-algebraic[22])
    rates[17] = custom_piecewise([greater(states[14] , 0.200000), (-constants[3]*((algebraic[24]-algebraic[41])+algebraic[37]))/(2.00000*constants[54]*constants[2]) , greater(states[15] , 0.0100000) & less_equal(states[14] , 0.200000), 0.00000 , True, -0.500000*states[17]])
    algebraic[20] = log(constants[13]/states[1])/constants[43]
    algebraic[21] = constants[12]*(power(states[2], 3.00000))*states[3]*states[4]*(states[0]-algebraic[20])
    algebraic[38] = constants[47]*(states[0]-algebraic[20])
    rates[1] = (-1.00000*constants[3]*(algebraic[21]+algebraic[38]+algebraic[41]*3.00000+algebraic[40]*3.00000))/(constants[54]*constants[2])
    algebraic[42] = algebraic[21]+algebraic[24]+algebraic[26]+algebraic[31]+algebraic[28]+algebraic[36]+algebraic[41]+algebraic[40]+algebraic[38]+algebraic[37]+algebraic[10]
    algebraic[44] = custom_piecewise([equal(constants[9] , 1.00000) & (less_equal(voi , 500.000) | greater(voi , 800.000)), (constants[10]-states[0])/1.00000 , equal(constants[9] , 1.00000) & greater(voi , 500.000) & less_equal(voi , 800.000), (constants[11]-states[0])/1.00000 , True, -1.00000*algebraic[42]])
    rates[0] = algebraic[44]
    rates[14] = custom_piecewise([greater(algebraic[44] , 150.000), 100.000*(1.00000-states[14])-0.500000*states[14] , True, -0.500000*states[14]])
    algebraic[43] = custom_piecewise([greater(states[17] , constants[29]), ((1.00000*constants[26]*(states[17]-constants[29]))/((constants[28]+states[17])-constants[29]))*(1.00000-states[15])*states[15] , less_equal(states[17] , constants[29]) & greater(states[19] , 0.00000), 0.00000*constants[27]*(1.00000-states[19])*states[19] , True, 0.00000])
    algebraic[45] = algebraic[43]*(states[13]-states[5])
    algebraic[46] = (constants[51]*states[5])/(states[5]+constants[33])
    algebraic[47] = constants[52]*states[21]
    algebraic[48] = 1.00000/(1.00000+constants[39]*((constants[38]*constants[36])/(power(constants[36]+states[5], 2.00000))+(constants[37]*constants[35])/(power(constants[35]+states[5], 2.00000))))
    rates[5] = algebraic[48]*((-constants[3]*((algebraic[24]-2.00000*algebraic[41])+algebraic[37]))/(2.00000*constants[54]*constants[2])+(algebraic[45]*constants[56])/constants[54]+((algebraic[47]-algebraic[46])*constants[55])/constants[54])
    algebraic[49] = (1.00000*(states[21]-states[13]))/constants[34]
    rates[21] = -1.00000*((algebraic[47]+(constants[56]/constants[55])*algebraic[49])-algebraic[46])
    algebraic[50] = 1.00000/(1.00000+(constants[40]*constants[30])/(power(constants[30]+states[13], 2.00000)))
    rates[13] = algebraic[50]*(algebraic[49]-algebraic[45])
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[1] = custom_piecewise([greater(fabs(states[0]+47.1300) , 0.00100000), (0.320000*(states[0]+47.1300))/(1.00000-exp(-0.100000*(states[0]+47.1300))) , True, 3.20000])
    algebraic[11] = 0.0800000*exp(-states[0]/11.0000)
    algebraic[2] = custom_piecewise([less(states[0] , -40.0000), 0.135000*exp((80.0000+states[0])/-6.80000) , True, 0.00000])
    algebraic[12] = custom_piecewise([less(states[0] , -40.0000), 3.56000*exp(0.0790000*states[0])+310000.*exp(0.350000*states[0]) , True, 1.00000/(0.130000*(1.00000+exp(-(states[0]+10.6600)/11.1000)))])
    algebraic[3] = custom_piecewise([less(states[0] , -40.0000), ((-127140.*exp(0.244000*states[0])-3.47400e-05*exp(-0.0439100*states[0]))*(states[0]+37.7800))/(1.00000+exp(0.311000*(states[0]+79.2300))) , True, 0.00000])
    algebraic[13] = custom_piecewise([less(states[0] , -40.0000), (0.121200*exp(-0.0105200*states[0]))/(1.00000+exp(-0.137800*(states[0]+40.1400))) , True, (0.300000*exp(-2.53500e-07*states[0]))/(1.00000+exp(-0.100000*(states[0]+32.0000)))])
    algebraic[4] = (14.9859/(16.6813*(power(2.00000* pi, 1.0/2))))*exp(-(power((states[0]-22.3600)/16.6813, 2.00000))/2.00000)
    algebraic[14] = 0.147100-(5.30000/(14.9300*(power(2.00000* pi, 1.0/2))))*exp(-(power((states[0]-6.27440)/14.9300, 2.00000))/2.00000)
    algebraic[5] = 0.00687200/(1.00000+exp((states[0]-6.15460)/6.12230))
    algebraic[15] = (0.0687000*exp(-0.108100*(states[0]+9.82550))+0.0112000)/(1.00000+exp(-0.277900*(states[0]+9.82550)))+0.000547400
    algebraic[6] = (0.526600*exp(-0.0166000*(states[0]-42.2912)))/(1.00000+exp(-0.0943000*(states[0]-42.2912)))
    algebraic[16] = (5.18600e-05*states[0]+0.514900*exp(-0.134400*(states[0]-5.00270)))/(1.00000+exp(-0.134800*(states[0]-5.18600e-05)))
    algebraic[7] = (5.61200e-05*states[0]+0.0721000*exp(-0.173000*(states[0]+34.2531)))/(1.00000+exp(-0.173200*(states[0]+34.2531)))
    algebraic[17] = (0.000121500*states[0]+0.0767000*exp(-1.66000e-09*(states[0]+34.0235)))/(1.00000+exp(-0.160400*(states[0]+34.0235)))
    algebraic[8] = 0.00301300/(1.00000+exp((7.44540-(states[0]+10.0000))/14.3171))
    algebraic[18] = 0.00587000/(1.00000+exp((5.95000+states[0]+10.0000)/15.8200))
    algebraic[9] = (0.00500000*exp(0.000526600*(states[0]+4.06700)))/(1.00000+exp(-0.126200*(states[0]+4.06700)))
    algebraic[19] = (0.0160000*exp(0.00160000*(states[0]+65.6600)))/(1.00000+exp(0.0783000*(states[0]+65.6600)))
    algebraic[25] = log((0.0430000*constants[13]+constants[17])/(0.0430000*states[1]+states[8]))/constants[43]
    algebraic[26] = constants[44]*states[10]*states[9]*(states[0]-algebraic[25])
    algebraic[30] = 1.00000/(1.00000+exp((states[0]+26.0000)/23.0000))
    algebraic[29] = log(constants[17]/states[8])/constants[43]
    algebraic[31] = constants[19]*states[12]*algebraic[30]*(states[0]-algebraic[29])
    algebraic[27] = log((0.0183300*constants[13]+constants[17])/(0.0183300*states[1]+states[8]))/constants[43]
    algebraic[28] = constants[18]*(power(states[11], 2.00000))*(states[0]-algebraic[27])
    algebraic[32] = log(constants[17]/states[8])/constants[43]
    algebraic[33] = 0.100000/(1.00000+exp(0.0600000*(states[0]-(algebraic[32]+200.000))))
    algebraic[34] = (3.00000*exp(0.000200000*(states[0]+100.000+-algebraic[32]))+1.00000*exp(0.100000*(states[0]-(10.0000+algebraic[32]))))/(1.00000+exp(-0.500000*(states[0]-algebraic[32])))
    algebraic[35] = algebraic[33]/(algebraic[33]+algebraic[34])
    algebraic[36] = constants[45]*algebraic[35]*(states[0]-algebraic[32])
    algebraic[39] = 1.00000/(1.00000+0.124500*exp(-0.100000*states[0]*constants[43])+0.0365000*constants[49]*exp(-states[0]*constants[43]))
    algebraic[40] = (((constants[48]*algebraic[39]*1.00000)/(1.00000+power(constants[20]/states[1], 1.50000)))*constants[17])/(constants[17]+constants[21])
    algebraic[0] = floor(voi/constants[6])*constants[6]
    algebraic[10] = custom_piecewise([greater_equal(voi-algebraic[0] , constants[5]) & less_equal(voi-algebraic[0] , constants[5]+constants[7]), constants[8] , True, 0.00000])
    algebraic[22] = log(constants[15]/states[5])/(2.00000*constants[43])
    algebraic[23] = constants[16]/(constants[16]+states[5])
    algebraic[24] = constants[14]*states[6]*states[7]*algebraic[23]*(states[0]-algebraic[22])
    algebraic[41] = ((((((constants[50]*1.00000)/(power(constants[22], 3.00000)+power(constants[13], 3.00000)))*1.00000)/(constants[23]+constants[15]))*1.00000)/(1.00000+constants[24]*exp((constants[25]-1.00000)*states[0]*constants[43])))*(exp(constants[25]*states[0]*constants[43])*(power(states[1], 3.00000))*constants[15]-exp((constants[25]-1.00000)*states[0]*constants[43])*(power(constants[13], 3.00000))*states[5])
    algebraic[37] = constants[46]*(states[0]-algebraic[22])
    algebraic[20] = log(constants[13]/states[1])/constants[43]
    algebraic[21] = constants[12]*(power(states[2], 3.00000))*states[3]*states[4]*(states[0]-algebraic[20])
    algebraic[38] = constants[47]*(states[0]-algebraic[20])
    algebraic[42] = algebraic[21]+algebraic[24]+algebraic[26]+algebraic[31]+algebraic[28]+algebraic[36]+algebraic[41]+algebraic[40]+algebraic[38]+algebraic[37]+algebraic[10]
    algebraic[44] = custom_piecewise([equal(constants[9] , 1.00000) & (less_equal(voi , 500.000) | greater(voi , 800.000)), (constants[10]-states[0])/1.00000 , equal(constants[9] , 1.00000) & greater(voi , 500.000) & less_equal(voi , 800.000), (constants[11]-states[0])/1.00000 , True, -1.00000*algebraic[42]])
    algebraic[43] = custom_piecewise([greater(states[17] , constants[29]), ((1.00000*constants[26]*(states[17]-constants[29]))/((constants[28]+states[17])-constants[29]))*(1.00000-states[15])*states[15] , less_equal(states[17] , constants[29]) & greater(states[19] , 0.00000), 0.00000*constants[27]*(1.00000-states[19])*states[19] , True, 0.00000])
    algebraic[45] = algebraic[43]*(states[13]-states[5])
    algebraic[46] = (constants[51]*states[5])/(states[5]+constants[33])
    algebraic[47] = constants[52]*states[21]
    algebraic[48] = 1.00000/(1.00000+constants[39]*((constants[38]*constants[36])/(power(constants[36]+states[5], 2.00000))+(constants[37]*constants[35])/(power(constants[35]+states[5], 2.00000))))
    algebraic[49] = (1.00000*(states[21]-states[13]))/constants[34]
    algebraic[50] = 1.00000/(1.00000+(constants[40]*constants[30])/(power(constants[30]+states[13], 2.00000)))
    return algebraic

def custom_piecewise(cases):
    """Compute result of a piecewise function"""
    return select(cases[0::2],cases[1::2])

def solve_model():
    """Solve model with ODE solver"""
    from scipy.integrate import ode
    # Initialise constants and state variables
    (init_states, constants) = initConsts()

    # Set timespan to solve over
    voi = linspace(0, 10, 500)

    # Construct ODE object to solve
    r = ode(computeRates)
    r.set_integrator('vode', method='bdf', atol=1e-06, rtol=1e-06, max_step=1)
    r.set_initial_value(init_states, voi[0])
    r.set_f_params(constants)

    # Solve model
    states = array([[0.0] * len(voi)] * sizeStates)
    states[:,0] = init_states
    for (i,t) in enumerate(voi[1:]):
        if r.successful():
            r.integrate(t)
            states[:,i+1] = r.y
        else:
            break

    # Compute algebraic variables
    algebraic = computeAlgebraic(constants, states, voi)
    return (voi, states, algebraic)

def plot_model(voi, states, algebraic):
    """Plot variables against variable of integration"""
    import numpy as np
    import pandas as pd
    import os
    (legend_states, legend_algebraic, legend_voi, legend_constants) = createLegends()
    results = np.vstack([voi[np.newaxis], states])
    cols = ['t', *legend_states]
    df = pd.DataFrame(results.T, columns=cols)
    data_dir = './cardiac-model/data/'
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    df.to_csv(os.path.join(data_dir, 'data.csv'))

    # import pylab
    # pylab.figure(1)
    # pylab.plot(voi,vstack((states,algebraic)).T)
    # pylab.xlabel(legend_voi)
    # pylab.legend(legend_states + legend_algebraic, loc='best')
    # pylab.show()

if __name__ == "__main__":
    (voi, states, algebraic) = solve_model()
    plot_model(voi, states, algebraic)