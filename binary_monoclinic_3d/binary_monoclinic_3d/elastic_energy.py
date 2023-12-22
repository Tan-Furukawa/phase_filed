import cupy as cp
import multiprocess as mp
import time

def get_i_j_index(i):
    if (i==0): return (0,0) 
    elif(i==1): return (1,1)
    elif(i==2): return (2,2)
    elif(i==3): return (1,2)
    elif(i==4): return (0,2)
    elif(i==5): return (0,1)
    else: raise TypeError("i is 0 to 5")

def target_process(results, fn, *args):
    results.put(fn(*args))

def solve_elasticity(tmatx, cm, c_p, ea, ei0, con, c0):
    Nx, Ny, Nz = con.shape

    niter = 10
    tolerance = 0.001
    old_norm = None

    # initialize stress
    s = cp.zeros((Nx,Ny,Nz,6))

    # initialize total strain
    e = cp.zeros((Nx,Ny,Nz,6))

    # elastic strain
    el = cp.zeros((Nx,Ny,Nz,6))

    #--- eigenstrains:
    # ei * c(x) where c(x) is concentration of precipitants at point x
    ei = cp.zeros((Nx, Ny, Nz, 6))
    for i in range(6):
        ei[:,:,:,i] = ei0[i] * (con - c0)

    # calculate effective elastic constants
    # effective elastic constant:
    # C = c(r)Cp + (1-c(r))Cm = (Cm + Cp)/2 + 1/2 * (1 - c(r))(Cp - Cm)
    c = cp.zeros((Nx,Ny,Nz,6,6))
    ci = c_p - cm
    I, J = ci.shape
    for i in range(I):
        for j in range(J):
            c[:,:,:,i,j] = con * c_p[i,j] + (1-con) * cm[i,j]

    #-- Green operator:
    # e[k] = e[k] - Γ:s[k]

    def get_ematx(ii, jj, ematx, smatx):
        return ematx

    def update_elastic_strain(index, e, smatx):
        I, J = get_i_j_index(index)
        e = cp.fft.fftn(e)
        for kk in range(3):
            for ll in range(3):
                    e = e - tmatx[:, :, :, I, J, kk, ll] * smatx[:, :, :, kk, ll]

        e = cp.real(cp.fft.ifftn(e))
        return index, e

    # ematx = cp.zeros((Nx, Ny, Nz, 3, 3), dtype=cp.complex128)
    smatx = cp.zeros((Nx, Ny, Nz, 3, 3), dtype=cp.complex128)
    ek = cp.zeros((Nx, Ny, Nz))

    for iter in range(niter):

        #--- take the stresses & strains to Fourier space
        smatx[:, :, :, 0, 0] = cp.fft.fftn(s[:,:,:,0])
        smatx[:, :, :, 1, 1] = cp.fft.fftn(s[:,:,:,1])
        smatx[:, :, :, 2, 2] = cp.fft.fftn(s[:,:,:,2])
        smatx[:, :, :, 1, 2] = cp.fft.fftn(s[:,:,:,3])
        smatx[:, :, :, 2, 1] = smatx[:, :, :, 1, 2]
        smatx[:, :, :, 0, 2] = cp.fft.fftn(s[:,:,:,4])
        smatx[:, :, :, 2, 0] = smatx[:, :, :, 0, 2]
        smatx[:, :, :, 0, 1] = cp.fft.fftn(s[:,:,:,5])
        smatx[:, :, :, 1, 0] = smatx[:, :, :, 0, 1]

        for i in range(6):
            I,J = get_i_j_index(i)
            ek = cp.fft.fftn(e[:,:,:,i])
            for kk in range(3):
                for ll in range(3):
                        ek = ek - tmatx[:, :, :, I, J, kk, ll] * smatx[:, :, :, kk, ll]
                        e[:,:,:,i] = cp.real(cp.fft.ifftn(ek))

        # parallelization
        # with mp.Manager() as manager:
        #     results = manager.Queue()
        #     processes = []

        #     for index in range(6):
        #         # update_elastic_strain(index, e, smatx)
        #         process = mp.Process(target=target_process, args=(results, update_elastic_strain, index, e[:,:,:,index], smatx))
        #         processes.append(process)
        #         process.start()

        #     for process in processes:
        #         process.join()

        #     while not results.empty():
        #         i, e[:,:,:,i] = results.get()

        # el: elastic strain
        #! ea反映してない ea - e - ei

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
        sum_stres = s[:,:,:,0] + s[:,:,:,1] + s[:,:,:,2] + s[:,:,:,3] + s[:,:,:,4] + s[:,:,:,5]
        normF = cp.linalg.norm(sum_stres)

        if iter != 0:
            conver = abs((normF - old_norm) / old_norm)
            if conver <= tolerance:
                break
        old_norm = normF

    #--- strain energy:
    # et: elastic strain components: # 名前わかりにくい！
    # e: total strain
    #! ea反映してない
    #! et11 = ea[0] + e11 - ei11
    el = e - ei

    # del E / del c
    # E = Cijkl * et_ij * et_kl
    # C = 1/2 (Cp + Cm) - (1/2 - c)(Cp - Cm), del C /del c = Cp - Cm
    # del E / del c = (Cp - Cm) * et_ij * et_kl - 2ei0 * Cijkl * et_kl * δ_ij
    # del et / del c = ei0はあってる？弾性ひずみは濃度の関数ではない？

    ci = c_p - cm

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

    # el = 0.5 * (c11*et11**2 + 2*c12*et11*et22 + 4*c15*et11*et12 + c22*et22**2 + 4*c25*et12*et22 + 4*c44*et12**2)

    return (delsdc0, s, el)
