import numpy as np
import scipy.io 

def angle2M_2(dipole):
    mux = np.sin(dipole[0]/180*np.math.pi)*np.cos(dipole[1]/180*np.math.pi)
    muy = np.sin(dipole[0]/180*np.math.pi)*np.sin(dipole[1]/180*np.math.pi)
    muz = np.cos(dipole[0]/180*np.math.pi)
    gamma = dipole[2]
    #gamma =  (tf.math.cos(dipole[2])**2+tf.math.cos(dipole[2]))/2
    
    
    M2 = [gamma*mux**2+(1-gamma)/3,gamma*muy**2+(1-gamma)/3,gamma*muz**2+(1-gamma)/3,gamma*mux*muy,gamma*mux*muz,gamma*muy*muz]
    M2 = np.reshape(M2,[1,6])
    return M2


def CRB_calculation3(Basis_matrix,dipole,signal,background):
    #dipole=[thetaD,phiD,gamma]
    d_theta = 10**(-6)
    d_phi = 10**(-6)
    d_gamma = 10**(-6)

    d_theta_m = np.array([d_theta,0,0.])
    d_phi_m = np.array([0,d_phi,0.])
    d_gamma_m = np.array([0,0,d_gamma])       
    #1
    dipole1 = dipole
    M_2 = angle2M_2(dipole1)
    #temp2 = tf.constant([1],dtype=tf.float64)
    I = signal*M_2@Basis_matrix+background
    #2
    dipole2 = dipole+d_phi_m
    M_2 = angle2M_2(dipole2)
    #temp2 = tf.constant([1],dtype=tf.float64)
    I_dTheta = signal*M_2@Basis_matrix+background


    #3
    dipole3 = dipole+d_theta_m
    M_2 = angle2M_2(dipole3)
    #temp2 = tf.constant([1],dtype=tf.float64)
    I_dPhi = signal*M_2@Basis_matrix+background
    
    #4
    dipole4 = dipole+d_gamma_m
    M_2 = angle2M_2(dipole4)
    #temp2 = tf.constant([1],dtype=tf.float64)
    I_dGamma = signal*M_2@Basis_matrix+background

    I_grad_theta = ((I_dTheta-I)/d_theta)
    I_grad_phi = ((I_dPhi-I)/d_phi)
    I_grad_gamma = ((I_dGamma-I)/d_gamma)

    FI11 = np.sum(I_grad_theta*I_grad_theta/I) + 1e-10
    FI22 = np.sum(I_grad_phi*I_grad_phi/I) + 1e-10
    FI33 = np.sum(I_grad_gamma*I_grad_gamma/I)+ 1e-10
    FI12 = np.sum(I_grad_phi*I_grad_theta/I)
    FI13 = np.sum(I_grad_theta*I_grad_gamma/I) + 1e-10
    FI23 = np.sum(I_grad_phi*I_grad_gamma/I) + 1e-10

    
    FI = ([[FI11,FI12,FI13],
            [FI12,FI22,FI23],
            [FI13,FI23,FI33]])
#                 FI = ([[FI22,FI23],
#                        [FI23,FI33]])
    #set_trace()
    CRB = np.linalg.inv(FI)
    sigma_thetaD = np.sqrt(CRB[0,0])
    sigma_phiD = np.sqrt(CRB[1,1])
    sigma_gamma = np.sqrt(CRB[2,2])

    return  sigma_thetaD,sigma_phiD,sigma_gamma


folder = '/home/wut/Documents/Deep-SMOLM/code_pytorch/DeepSMOLM/forward_model_pixOL/'
basis_matrix_opt = scipy.io.loadmat(folder+'basis_matrix_opt.mat')
basis_matrix_opt = np.transpose(basis_matrix_opt['basis_matrix_opt'])
dipole = np.array([60,180,1])  #dipole=[thetaD,phiD,gamma]
signal=1000
background = 2
[sigma_thetaD,sigma_phiD,sigma_gamma]=CRB_calculation3(basis_matrix_opt,dipole,signal,background)



