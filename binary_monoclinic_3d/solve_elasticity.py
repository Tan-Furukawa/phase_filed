import numpy as np
import multiprocessing
import time

def solve_elasticity(Nx, Ny, Nz, tmatx, cm, cp, ea, ei0, con, c0):
    # np.set_printoptions(precision=15)

    niter = 10
    tolerance = 0.001
    old_norm = None

    # initialize stress
    s = np.zeros((Nx,Ny,Nz,6))

    # initialize strain
    e = np.zeros((Nx,Ny,Nz,6))

    # elastic strain
    el = np.zeros((Nx,Ny,Nz,6))

    #--- eigenstrains:
    # ei * c(x) where c(x) is concentration of precipitants at point x

    ei = np.zeros((Nx, Ny, Nz, 6))
    ei[:,:,:,0] = ei0[0] * (con - c0)
    ei[:,:,:,1] = ei0[1] * (con - c0)
    ei[:,:,:,2] = ei0[2] * (con - c0)
    ei[:,:,:,3] = ei0[3] * (con - c0)
    ei[:,:,:,4] = ei0[4] * (con - c0)
    ei[:,:,:,5] = ei0[5] * (con - c0)
    # ei = 
    # ei11, ei12, ei13 = ei011 * (con-c0), ei012 * (con-c0), ei013 * (con-c0)
    # ei22, ei23 = ei022 * (con-c0), ei023 * (con-c0)
    # ei33 = ei033 * (con-c0)

    # calculate effective elastic constants
    # effective elastic constant: 
    # C = c(r)Cp + (1-c(r))Cm = (Cm + Cp)/2 + 1/2 * (1 - c(r))(Cp - Cm)
    # cubic elastic constant
    # 1111 = 11 # 1122 = 12 # 2222 = 22 # 1212 = 44
    # C =  c11 c12  0 
    #      c12 c11  0 
    #       0   0  c44

    c = np.zeros((Nx,Ny,Nz,6,6))
    ci = cp - cm
    I, J = ci.shape
    for i in range(I):
        for j in range(J):
            c[:,:,:,i,j] = con * cp[i,j] + (1-con) * cm[i,j]

    #-- Green operator:
    # e[k] = e[k] - Γ:s[k]
    def get_ematx (ii, jj, ematx, smatx):
        for kk in range(3):
            for ll in range(3):
                    ematx = ematx - tmatx[:, :, :, ii, jj, kk, ll] * smatx[:, :, :, kk, ll]
        return ematx

    def get_i_j_index(i):
        if (i==0): return (0,0) 
        elif(i==1): return (1,1)
        elif(i==2): return (2,2)
        elif(i==3): return (1,2)
        elif(i==4): return (0,2)
        elif(i==5): return (0,1)
        else: raise TypeError("i is 0 to 5")

    ematx = np.zeros((Nx, Ny, Nz, 3, 3), dtype=np.complex128)
    smatx = np.zeros((Nx, Ny, Nz, 3, 3), dtype=np.complex128)

    for iter in range(niter):

        #--- take the stresses & strains to Fourier space

        smatx[:, :, :, 0, 0] = np.fft.fftn(s[:,:,:,0])
        smatx[:, :, :, 1, 1] = np.fft.fftn(s[:,:,:,1])
        smatx[:, :, :, 2, 2] = np.fft.fftn(s[:,:,:,2])
        smatx[:, :, :, 1, 2] = np.fft.fftn(s[:,:,:,3])
        smatx[:, :, :, 2, 1] = np.fft.fftn(s[:,:,:,3])
        smatx[:, :, :, 0, 2] = np.fft.fftn(s[:,:,:,4])
        smatx[:, :, :, 2, 0] = np.fft.fftn(s[:,:,:,4])
        smatx[:, :, :, 0, 1] = np.fft.fftn(s[:,:,:,5])
        smatx[:, :, :, 1, 0] = np.fft.fftn(s[:,:,:,5])

        # strain
        # print("-----------1-------------")
        g = time.time()

        # summarize elastic strain tensor
        for i in range(6):
            I, J = get_i_j_index(i)
            ematx[:, :, :, I, J] = np.fft.fftn(e[:,:,:,i])
            e[:,:,:,i] = np.real(np.fft.ifftn(get_ematx(I, J, ematx[:, :, :, I, J], smatx)))

        # ematx[:, :, :, 1, 1] = np.fft.fftn(e[:,:,:,1])
        # e[:,:,:,1] = np.real(np.fft.ifftn(get_ematx(1, 1, ematx[:, :, :, 1, 1], smatx)))

        # ematx[:, :, :, 2, 2] = np.fft.fftn(e[:,:,:,2])
        # e[:,:,:,2] = np.real(np.fft.ifftn(get_ematx(2, 2, ematx[:, :, :, 2, 2], smatx)))

        # ematx[:, :, :, 1, 2] = np.fft.fftn(e[:,:,:,3])
        # e[:,:,:,3] = np.real(np.fft.ifftn(get_ematx(1, 2, ematx[:, :, :, 1, 2], smatx)))

        # ematx[:, :, :, 0, 2] = np.fft.fftn(e[:,:,:,4])
        # e[:,:,:,4] = np.real(np.fft.ifftn(get_ematx(0, 2, ematx[:, :, :, 0, 2], smatx)))

        # ematx[:, :, :, 0, 1] = np.fft.fftn(e[:,:,:,5])
        # e[:,:,:,5] = np.real(np.fft.ifftn(get_ematx(0, 1, ematx[:, :, :, 0, 1], smatx)))

        # print(time.time() - g)

        # el: elastic strain
        #! ea反映してない ea - e - ei
        # print("-----------5-------------")
        g = time.time()
        el = e - ei
        # el33 = ea[2,2] + e33 - ei33

        # Calculate stresses:
        # ea: applied strain
        # e:  fluctuation strain
        # ei: eigen strain
        # s = C * (ea + e - ei)
        #! ea反映してない
        #! 例:s11 = c11 * (ea[0] + e11 - ei11) + c12 * (ea[1] + e22 - ei22) + 2.0 * c16 * (ea[2] + e12 - ei12)
        s[:,:,:,0] = c[:,:,:,0, 0]*el[:,:,:,0] + c[:,:,:,0, 1]*el[:,:,:,1] + c[:,:,:,0, 2]*el[:,:,:,2] + c[:,:,:,0, 4]*el[:,:,:,4]
        s[:,:,:,1] = c[:,:,:,0, 1]*el[:,:,:,0] + c[:,:,:,1, 1]*el[:,:,:,1] + c[:,:,:,1, 2]*el[:,:,:,2] + c[:,:,:,1, 4]*el[:,:,:,4]
        s[:,:,:,2] = c[:,:,:,0, 2]*el[:,:,:,0] + c[:,:,:,1, 2]*el[:,:,:,1] + c[:,:,:,2, 2]*el[:,:,:,2] + c[:,:,:,2, 4]*el[:,:,:,4]
        s[:,:,:,3] = c[:,:,:,3, 3]*el[:,:,:,3] + c[:,:,:,3, 5]*el[:,:,:,5]
        s[:,:,:,4] = c[:,:,:,0, 4]*el[:,:,:,0] + c[:,:,:,1, 4]*el[:,:,:,1] + c[:,:,:,2, 4]*el[:,:,:,2] + c[:,:,:,4, 4]*el[:,:,:,4]
        s[:,:,:,5] = c[:,:,:,3, 5]*el[:,:,:,3] + c[:,:,:,5, 5]*el[:,:,:,5]

        # s12 = c14 * (ea[2] + e11 - ei11) + c24 * (ea[1] + e22 - ei22) + 2.0 * c44 * (ea[2] + e12 - ei12)
        # s22 = c12 * (ea[0] + e11 - ei11) + c22 * (ea[1] + e22 - ei22) + 2.0 * c24 * (ea[2] + e12 - ei12)
        # s12 = 2.0 * c44 * (ea[2] + e12 - ei12)

        #---check convergence:
        # print("-----------6-------------")
        sum_stres = s[:,:,:,0] + s[:,:,:,1] + s[:,:,:,2] + s[:,:,:,3] + s[:,:,:,4] + s[:,:,:,5]
        normF = np.linalg.norm(sum_stres)

        if iter != 0:
            conver = abs((normF - old_norm) / old_norm)
            print(conver)
            if conver <= tolerance:
                break
        old_norm = normF
        # print(time.time() - g)

    #--- strain energy:
    # et: elastic strain components: # 名前わかりにくい！
    # e: total strain
    #! ea反映してない
    #! et11 = ea[0] + e11 - ei11
    el = e - ei
    # et11 = e11 - ei11
    # et12 = e12 - ei12
    # et13 = e13 - ei13
    # et22 = e22 - ei22
    # et23 = e23 - ei23
    # et33 = e33 - ei33

    # del E / del c
    # E = Cijkl * et_ij * et_kl
    # C = 1/2 (Cp + Cm) - (1/2 - c)(Cp - Cm), del C /del c = Cp - Cm
    # del E / del c = (Cp - Cm) * et_ij * et_kl - 2ei0 * Cijkl * et_kl * δ_ij

    # del et / del c = ei0はあってる？弾性ひずみは濃度の関数ではない？
    ci = cp - cm
    delsdc0 = 0.5*(
       -(2*c[:,:,:,3, 3]*ei0[3] + 2*c[:,:,:,3, 5]*ei0[5])*el[:,:,:,3] -  \
        (2*c[:,:,:,3, 5]*ei0[3] + 2*c[:,:,:,5, 5]*ei0[5])*el[:,:,:,5] +  \
        (ci[3, 3]*el[:,:,:,3] + ci[3, 5]*el[:,:,:,5])*el[:,:,:,3] +  \
        (ci[3, 5]*el[:,:,:,3] + ci[5, 5]*el[:,:,:,5])*el[:,:,:,5] -  \
        (2*c[:,:,:,0, 0]*ei0[0] + 2*c[:,:,:,0, 1]*ei0[1] + 2*c[:,:,:,0, 2]*ei0[2] + 2*c[:,:,:,0, 4]*ei0[4])*el[:,:,:,0] -  \
        (2*c[:,:,:,0, 1]*ei0[0] + 2*c[:,:,:,1, 1]*ei0[1] + 2*c[:,:,:,1, 2]*ei0[2] + 2*c[:,:,:,1, 4]*ei0[4])*el[:,:,:,1] -  \
        (2*c[:,:,:,0, 2]*ei0[0] + 2*c[:,:,:,1, 2]*ei0[1] + 2*c[:,:,:,2, 2]*ei0[2] + 2*c[:,:,:,2, 4]*ei0[4])*el[:,:,:,2] -  \
        (2*c[:,:,:,0, 4]*ei0[0] + 2*c[:,:,:,1, 4]*ei0[1] + 2*c[:,:,:,2, 4]*ei0[2] + 2*c[:,:,:,4, 4]*ei0[4])*el[:,:,:,4] +  \
        (ci[0, 0]*el[:,:,:,0] + ci[0, 1]*el[:,:,:,1] + ci[0, 2]*el[:,:,:,2] + ci[0, 4]*el[:,:,:,4])*el[:,:,:,0] +  \
        (ci[0, 1]*el[:,:,:,0] + ci[1, 1]*el[:,:,:,1] + ci[1, 2]*el[:,:,:,2] + ci[1, 4]*el[:,:,:,4])*el[:,:,:,1] +  \
        (ci[0, 2]*el[:,:,:,0] + ci[1, 2]*el[:,:,:,1] + ci[2, 2]*el[:,:,:,2] + ci[2, 4]*el[:,:,:,4])*el[:,:,:,2] +  \
        (ci[0, 4]*el[:,:,:,0] + ci[1, 4]*el[:,:,:,1] + ci[2, 4]*el[:,:,:,2] + ci[4, 4]*el[:,:,:,4])*el[:,:,:,4]
    )


    # delsdc0 = 0.5 * (
    #     # 4*(-cm14 + cp14)*et11*et12 + 4*(-cm24 + cp24)*et12*et22 + 4*(-cm44 + cp44)*et12**2 \
    #     # - 2*c11*ei011*et11 - 2*c12*ei011*et22 - 2*c12*ei022*et11 \
    #     # - 4*c14*ei011*et12 - 4*c14*ei012*et11 - 2*c22*ei022*et22 - 4*c24*ei012*et22 - 4*c24*ei022*et12 \
    #     # - 8*c44*ei012*et12 + (cp11-cm11)*et11**2 + 2*(cp12-cm12)*et11*et22 + (cp22-cm22)*et22**2
    # )

    # el = 0.5 * (c11*et11**2 + 2*c12*et11*et22 + 4*c15*et11*et12 + c22*et22**2 + 4*c25*et12*et22 + 4*c44*et12**2)

    return (delsdc0, s, el)

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
