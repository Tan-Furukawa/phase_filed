#%%
# from matplotlib.colors import Normalize
import cupy as cp
import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
import binary_monoclinic_3d.aliasing as aliasing
from binary_monoclinic_3d.initial_noise import add_initial_noise
from binary_monoclinic_3d.prepare_fft import prepare_fft
from binary_monoclinic_3d.green_tensor import green_tensor
from binary_monoclinic_3d.elastic_energy import solve_elasticity
from binary_monoclinic_3d.free_energy import get_free_energy
import os
from binary_monoclinic_3d._plot import dim3_plot as myplt3
from binary_monoclinic_3d._save import save_3d_plot as save

class BinaryMonoclinic3D(object):

    def __init__(self, save_path, method="linear"):
        # the path of saved files
        self.method = method
        self.save_path = save_path
        self.dirname = save.make_dir_name()
        self.set_initial_parameters()

    def set_all(self):
        self.make_save_file()
        self.save_instance()
        self.make_calculation_parameters()
        self.prepare_result_variables()
        self.calculate_green_tensor()

    def doit(self):
        self.set_all()
        self.calculate_phase_filed(method = self.method)

    def set_initial_parameters(self):
        self.iter = 0
        self.roop_start_from = 1
        self.Nx = 64; self.Ny = 64; self.Nz = 64
        self.dx = 1.0; self.dy = 1.0; self.dz = 1.0
        self.nstep = 10000
        self.nsave = 10
        self.nprint = 100
        self.dtime = 5.0e-2
        self.coefA = 1.0
        self.c0 = 0.4
        self.mobility = 1.0
        self.grad_coef = 2.0
        self.noise = 0.1
        self.__R = 8.31446262
        self.P = 1 * 10**5 # [Pa]
        self.T = 600 # [K]
        self.w_or_input = lambda T, P: (22820 - 6.3 * T + 0.461 * P / 10 ** 5)  # [GPa]
        self.w_ab_input = lambda T, P: (19550 - 10.5 * T + 0.327 * P / 10 ** 5) # [GPa]
        self.v_or = 8.60 * 13.2  * 7.18 * np.sin(116 / 180 * np.pi) #[A^3]
        self.v_ab = 8.15 * 12.85 * 7.12 * np.sin(116 / 180 * np.pi) #[A^3]
        # Cij[GPa] * 10^9 * v[Å] * 10*(-30) * NA[/mol] = [/mol]
        self.cm_input = cp.array([
            [ 93.9,  41.5,  52.2,    0, -26.2,    0],
            [ 41.5, 176.8,  23.1,    0,  14.2,    0],
            [ 52.2,  23.1,  82.1,    0, -19.5,    0],
            [    0,     0,     0, 17.8,     0,  9.7],
            [-26.2,  14.2, -19.5,    0,  44.2,    0],
            [    0,     0,     0,  9.7,     0, 35.0]
        ])
        # con = cpのmol濃度
        self.cp_input = cp.array([
            [ 93.9,  41.5,  52.2,    0, -26.2,    0],
            [ 41.5, 176.8,  23.1,    0,  14.2,    0],
            [ 52.2,  23.1,  82.1,    0, -19.5,    0],
            [    0,     0,     0, 17.8,     0,  9.7],
            [-26.2,  14.2, -19.5,    0,  44.2,    0],
            [    0,     0,     0,  9.7,     0, 35.0]
        ])

        # applied strains
        self.ea = cp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # eigen strain (del e / del c)
        self.ei0 = cp.array([0.0543, 0.0115, 0.0110, 0, 0.0131, 0])
        # ei0 = np.array([0.0567, 0, 0.016858, 0, 0.016896, 0]) #Robin 1974

    def make_calculation_parameters (self):
        self.__wor = self.w_or_input(self.T, self.P) / (self.__R * self.T)
        self.__wab = self.w_ab_input(self.T, self.P) / (self.__R * self.T)

        N_A = 6.02 * 10.0 ** 23
        n_or = 1 / (self.v_or * 10**(-30)) * 4 / N_A
        n_ab = 1 / (self.v_ab * 10**(-30)) * 4 / N_A
        self.__n0 = n_or * self.c0 + (1 - self.c0) * n_ab

        self.__cm = self.cm_input * 10.0**9 / (self.__R * self.T) / self.__n0
        self.__cp = self.cp_input * 10.0**9 / (self.__R * self.T) / self.__n0

        self.__bulk = self.c0
        self.__kx, self.__ky, self.__kz, self.__k2, self.__k4, \
        self.__kx_mat, self.__ky_mat, self.__kz_mat = \
            prepare_fft(self.Nx, self.Ny, self.Nz, self.dx, self.dy, self.dz)

        # padding
        self.__pkx, self.__pky, self.__pkz, self.__pk2, self.__pk4, \
        self.__pkx_mat, self.__pky_mat, self.__pkz_mat = \
            prepare_fft(
                aliasing.get_padding_len(self.Nx) * 2 + self.Nx,
                aliasing.get_padding_len(self.Ny) * 2 + self.Ny,
                aliasing.get_padding_len(self.Nz) * 2 + self.Nz,
                self.dx, self.dy, self.dz)

    def make_save_file(self):
        save.create_directory(self.save_path)
        save.create_directory(f"{self.save_path}/{self.dirname}/res")

    def save_instance(self):
        res_dict = {}
        for key in self.__dict__:
            typename = type(self.__dict__[key]).__name__
            if (typename != "function"):
                res_dict[key] = self.__dict__[key]

        with open(f"{self.save_path}/{self.dirname}/instance.pickle", 'wb') as file:
            pickle.dump(res_dict, file)

    def load_instance(self, full_dir_path=None):
        if full_dir_path is None:
            with open(f"{self.save_path}/{self.dirname}/instance.pickle", 'rb') as file:
                loaded_instance = pickle.load(file)
            return loaded_instance
        else:
            with open(f"{full_dir_path}/instance.pickle", 'rb') as file:
                loaded_instance = pickle.load(file)
            return loaded_instance

    def prepare_result_variables(self):

        self.energy_g = cp.zeros(self.nstep) + cp.nan
        self.energy_el = cp.zeros(self.nstep) + cp.nan

        # derivatives of elastic energy
        self.delsdc = cp.zeros((self.Nx, self.Ny, self.Nz))
        # derivatives of free energy
        self.dfdcon = cp.zeros((self.Nx, self.Ny, self.Nz))
        # free energy
        self.g = cp.zeros((self.Nx, self.Ny, self.Nz))
        # elastic stress
        self.s = cp.zeros((self.Nx, self.Ny, self.Nz, 6))
        # elastic strain
        self.el = cp.zeros((self.Nx, self.Ny, self.Nz, 6))
        self.conk = cp.zeros((self.Nx, self.Ny, self.Nz, 6), dtype=cp.complex128)
        self.dgdck = cp.zeros((self.Nx, self.Ny, self.Nz, 6), dtype=cp.complex128)

        # set initial compositional noise
        self.con = add_initial_noise(self.Nx, self.Ny, self.Nz, self.c0, self.noise)

        cp.random.seed(123)

    def is_included_target_file_in_directory(self, directory, target_name):
        for root, _, files in os.walk(directory):
            for file_name in files:
                if target_name == file_name:
                    return True
        return False

    def calculate_green_tensor(self):
        # the calculation of green tensor costs very high.
        # so save the green tensor result and
        # use previous one if once it is calculated.
        print(f"parameters: Nx{self.Nx}, T{int(self.T)}K, P{int(self.P)}Pa")
        save.create_directory("resources")
        filename = f"tmatx_3d_feldspar_{int(self.Nx)}_{int(self.T)}K_{self.P}Pa.npy"

        if (self.is_included_target_file_in_directory("resources", filename)):
            print("using previous tmatx")
            self.tmatx = cp.asarray(np.load(f"resources/{filename}"))
        else:
            print("calculating tmatx")
            tmatx, omeg11 = green_tensor(
                cp.asnumpy(self.__kx),cp.asnumpy(self.__ky),cp.asnumpy(self.__kz),
                cp.asnumpy(self.__cp),cp.asnumpy(self.__cm))
            np.save(f"resources/{filename}", tmatx)
            self.tmatx = cp.asarray(tmatx)

    def calculate_phase_filed(self, method = "linear"):
        # roop_range = tqdm.tqdm(range(1, self.nstep + 1))
        roop_range = range(self.roop_start_from, self.nstep + 1)
        for istep in roop_range:
                self.iter = istep
                # print(istep)
                # Calculate derivatives of free energy and elastic energy
                self.delsdc, self.s, self.el = solve_elasticity(
                    self.tmatx, self.__cm, self.__cp,
                    self.ea, self.ei0, self.con, self.c0
                    )


                # Assuming you have the get_free_energy and solve_elasticity_v2 functions
                self.dfdcon, self.g = get_free_energy(self.con, self.__wab, self.__wor)


                self.energy_g[istep-1] = cp.sum(self.g)

                self.conk = cp.fft.fftn(self.con)
                self.dgdck = cp.fft.fftn(self.dfdcon + self.delsdc)
                # self.delsdck = cp.fft.fftn(self.delsdc)

                if (method != "linear"):
                    # diffusion term
                    self.d = self.con * (1 - self.con)
                    self.dk = cp.fft.fftn(self.d)
                    term1 = \
                        cp.fft.fftn(
                            cp.fft.ifftn(aliasing.add_aliasing(1.0j * self.__kx_mat * self.dk)) * \
                            cp.fft.ifftn(aliasing.add_aliasing(1.0j * self.__kx_mat * self.dgdck))
                        ) + \
                        cp.fft.fftn(
                            cp.fft.ifftn(aliasing.add_aliasing(1.0j * self.__ky_mat * self.dk)) * \
                            cp.fft.ifftn(aliasing.add_aliasing(1.0j * self.__ky_mat * self.dgdck))
                        ) + \
                        cp.fft.fftn(
                            cp.fft.ifftn(aliasing.add_aliasing(1.0j * self.__kz_mat * self.dk)) * \
                            cp.fft.ifftn(aliasing.add_aliasing(1.0j * self.__kz_mat * self.dgdck))
                        )
                    self.gk = -self.__k2 * self.dgdck - self.__k4 * self.grad_coef * self.conk

                    term2 = cp.fft.fftn(
                        cp.fft.ifftn(aliasing.add_aliasing(self.gk)) *
                        cp.fft.ifftn(aliasing.add_aliasing(self.dk)))
                    t1_size = cp.sum(cp.abs(term1))
                    t2_size = cp.sum(cp.abs(term2))

                    print(t1_size / (t1_size + t2_size))

                    self.conk = self.conk + self.dtime * (
                        # term1 + term2
                        aliasing.remove_aliasing(term1, self.d.shape) + \
                        aliasing.remove_aliasing(term2, self.d.shape)
                    )
                    # self.con = cp.real(cp.fft.ifftn(self.conk))
                    self.con = cp.real(cp.fft.ifftn(self.conk))
                    # myplt3.display_3d_matrix(cp.asnumpy(self.con))
                    # , shape=self.d.shape
                    # aliasing.remove_aliasing(
                        # )
                    # myplt3.display_3d_matrix(cp.asnumpy(self.con))
                else:
                    # Time integration
                    numer = self.dtime * self.mobility * self.__k2 * (self.dgdck)
                    denom = 1.0 + self.dtime * self.coefA * self.mobility * self.grad_coef * self.__k4

                    self.conk = (self.conk - numer) / denom
                    self.con = cp.real(cp.fft.ifftn(self.conk))

                if (self.nprint is not None):
                    if (istep % self.nprint == 0) or (istep == 1):
                        # con_disp = np.flipud(cp.asnumpy(self.con.transpose()))
                        con_disp = self.con
                        # plt.imshow(con_disp)の図の向きは、
                        # y
                        # ↑
                        # |
                        # + --→ x [100]
                        # となる。
                        myplt3.display_3d_matrix(cp.asnumpy(con_disp))

                if (istep % self.nsave == 0) or (istep == 1):
                    np.save(f"{self.save_path}/{self.dirname}/res/con_{istep}.npy", cp.asnumpy(self.con))
                    np.save(f"{self.save_path}/{self.dirname}/res/el_{istep}.npy", cp.asnumpy(self.el))
                    np.save(f"{self.save_path}/{self.dirname}/res/s_{istep}.npy", cp.asnumpy(self.s))

if __name__ == "__main__":
    feldspar = BinaryMonoclinic3D("result", method="nonliear")
    # feldspar.doit()
    feldspar.dtime = 4e-3
    feldspar.nsave = 50
    feldspar.nprint = 10
    feldspar.nstep = 1000000
    # feldspar.set_all()
    feldspar.doit()
    # feldspar.roop_start_from = 279601
    # feldspar.con = cp.asarray(np.load("result/important/res/con_279600.npy"))
    # feldspar.calculate_phase_filed()

# %%
