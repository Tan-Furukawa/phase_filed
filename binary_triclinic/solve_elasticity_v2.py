import numpy as np

def solve_elasticity_v2(Nx, Ny, tmatx, cm, cp, ea, ei0, con):
    # np.set_printoptions(precision=15)

    niter = 10
    tolerance = 0.001
    old_norm = None

    # initialize stress
    s11 = np.zeros((Nx, Ny))
    s22 = np.zeros((Nx, Ny))
    s12 = np.zeros((Nx, Ny))

    # initialize strain
    e11 = np.zeros((Nx, Ny))
    e22 = np.zeros((Nx, Ny))
    e12 = np.zeros((Nx, Ny))

    #--- eigenstrains:
    # ei * c(x) where c(x) is concentration of precipitants at point x
    # ei = ei11 ei12 = ei11  0
    #      ei21 ei22    0   ei22
    ei011 = ei0[0,0]
    ei022 = ei0[1,1]
    ei012 = ei0[0,1]

    ei11 = ei0[0,0] * con
    ei22 = ei0[1,1] * con
    ei12 = ei0[0,1] * con

    # calculate effective elastic constants
    # effective elastic constant: 
    # C = c(r)Cp + (1-c(r))Cm = (Cm + Cp)/2 + 1/2 * (1 - c(r))(Cp - Cm)
    # cubic elastic constant
    # 1111 = 11 # 1122 = 12 # 2222 = 22 # 1212 = 44
    # C =  c11 c12  0   0  
    #      c12 c11  0   0 
    #       0   0  c44  0 
    #       0   0   0  c44
    cp11 = cp[0,0]
    cp12 = cp[0,1]
    cp22 = cp[1,1]
    cp14 = cp[0,2]
    cp24 = cp[1,2]
    cp44 = cp[2,2]

    cm11 = cm[0,0]
    cm12 = cm[0,1]
    cm22 = cm[1,1]
    cm14 = cm[0,2]
    cm24 = cm[1,2]
    cm44 = cm[2,2]

    c11 = con * cp[0,0] + (1.0 - con) * cm[0,0]
    c12 = con * cp[0,1] + (1.0 - con) * cm[0,1]
    c22 = con * cp[1,1] + (1.0 - con) * cm[1,1]
    c14 = con * cp[0,2] + (1.0 - con) * cm[0,2]
    c24 = con * cp[1,2] + (1.0 - con) * cm[1,2]
    c44 = con * cp[2,2] + (1.0 - con) * cm[2,2]

    for iter in range(niter):

        #--- take the stresses & strains to Fourier space
        # print(np.sum(s11))

        # strain
        e11k = np.fft.fft2(e11)
        e22k = np.fft.fft2(e22)
        e12k = np.fft.fft2(e12)

        # stress
        s11k = np.fft.fft2(s11)
        s22k = np.fft.fft2(s22)
        s12k = np.fft.fft2(s12)

        # summarize stress tensor
        smatx = np.zeros((Nx, Ny, 2, 2), dtype=np.complex128)
        smatx[:, :, 0, 0] = s11k
        smatx[:, :, 0, 1] = s12k
        smatx[:, :, 1, 0] = s12k
        smatx[:, :, 1, 1] = s22k

        # summarize elastic strain tensor
        ematx = np.zeros((Nx, Ny, 2, 2), dtype=np.complex128)
        ematx[:, :, 0, 0] = e11k
        ematx[:, :, 0, 1] = e12k
        ematx[:, :, 1, 0] = e12k
        ematx[:, :, 1, 1] = e22k

        #-- Green operator:
        # e[k] = e[k] - Γ:s[k]

        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    for ll in range(2):
                        ematx[:, :, ii, jj] = ematx[:, :, ii, jj] - tmatx[:, :, ii, jj, kk, ll] \
                            * smatx[:, :, kk, ll]

        e11k = ematx[:, :, 0, 0]
        e22k = ematx[:, :, 1, 1]
        e12k = ematx[:, :, 0, 1]

        # From Fourier space to real space:
        e11 = np.real(np.fft.ifft2(e11k))
        e22 = np.real(np.fft.ifft2(e22k))
        e12 = np.real(np.fft.ifft2(e12k))

        # Calculate stresses:
        # ea: applied strain
        # e:  fluctuation strain
        # ei: eigen strain

        # s = C * (ea + e - ei)

        s11 = c11 * (ea[0] + e11 - ei11) + c12 * (ea[1] + e22 - ei22)
        # print(ei11)
        # print(s11)
        # raise TypeError()
        s22 = c11 * (ea[1] + e22 - ei22) + c12 * (ea[0] + e11 - ei11)
        s12 = 2.0 * c44 * (ea[2] + e12 - ei12)

        #---check convergence:
        sum_stres = s11 + s22 + s12
        normF = np.linalg.norm(sum_stres, ord=2)
        if iter != 0:
            conver = abs((normF - old_norm) / old_norm)
            if conver <= tolerance:
                break
        old_norm = normF

    #--- strain energy:
    # et: elastic strain components:
    # e: total strain
    et11 = ea[0] + e11 - ei11
    et22 = ea[1] + e22 - ei22
    et12 = ea[2] + e12 - ei12

    # del E / del c
    # E = Cijkl * et_ij * et_kl
    # C = 1/2 (Cp + Cm) - (1/2 - c)(Cp - Cm), del C /del c = Cp - Cm
    # del E / del c = (Cp - Cm) * et_ij * et_kl - 2ei0 * Cijkl * et_kl * δ_ij

    # del et / del c = ei0はあってる？弾性ひずみは濃度の関数ではない？


    el = 0.5 * (et11 ** 2 * c11 + et22 ** 2 * c11 + 2 * et11 * c12 * et22 + 4 * et12 ** 2 * c44)
    # print(el)
    # raise TypeError()

    # delsdc0 = 0.5 * (et11 * ((cp12 - cm12) * et22 + (cp11 - cm11) * et11 - c12 * ei0 - c11 * ei0) \
    #     - ei0 * (c12 * et22 + c11 * et11) + ((cp11 - cm11) * et22 + (cp12 - cm12) * et11 - c12 * ei0 - c11 * ei0) * et22 \
    #     - ei0 * (c11 * et22 + c12 * et11) + 2.0 * (cp44 - cm44) * et12**2 - 4.0 * ei0 * c44 * et12)
    # delsdc0 = 0.5 * (et11 * ((cp12 - cm12) * et22 + (cp11 - cm11) * et11 - c12 * ei0 - c11 * ei0) \
    #     - ei0 * (c12 * et22 + c11 * et11) + ((cp11 - cm11) * et22 + (cp12 - cm12) * et11 - c12 * ei0 - c11 * ei0) * et22 \
    #     - ei0 * (c11 * et22 + c12 * et11) + 4.0 * (cp44 - cm44) * et12**2)

    # print(ei11)
    # print(np.sum(cp14))
    # print(np.sum(cm14))
    # print(np.sum(ei12))
    # print(np.sum(ei22 != ei11))
    # print(np.sum(c24))
    # print(np.sum(c14))
    # print(np.sum(cp24))
    # print(np.sum(cm24))
    # print(np.sum(c11 != c22))
    # print(np.sum(cp11 != cp22))
    # print(np.sum(cm11 != cm22))
    delsdc0 = 0.5 * (
        4*(-cm14 + cp14)*et11*et12 + 4*(-cm24 + cp24)*et12*et22 + 4*(-cm44 + cp44)*et12**2 \
        - 2*c11*ei011*et11 - 2*c12*ei011*et22 - 2*c12*ei022*et11 \
        - 4*c14*ei011*et12 - 4*c14*ei012*et11 - 2*c22*ei022*et22 - 4*c24*ei012*et22 - 4*c24*ei022*et12 \
        - 8*c44*ei012*et12 + (cp11-cm11)*et11**2 + 2*(cp12-cm12)*et11*et22 + (cp22-cm22)*et22**2
    )

    return (delsdc0, et11, et22, et12 ,s11, s22, s12, el)

# 1/2 * (
#     et11 * (
#         (cp12 - cm12) * et22 
#         + (cp11 - cm11) * et11 
#         - 2 * c12 * ei22
#         - 2 * c11 * ei11
#     )
#     + et22 * (
#         (cp11 - cm11) * et22 +
#          (cp12 - cm12) * et11 
#          - 2 * c12 * ei11
#          - 2 * c11 * ei
#     ) 
#     + 2.0 * (cp44 - cm44) * et12**2 
#     - 4.0 * ei0 * c44 * et12
# )
