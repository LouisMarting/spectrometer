import numpy as np

from ..physics import c0
from .transformations import abcd_seriesload,abcd_shuntload,Zin_from_abcd
from ..utils.utils import res_variance


class TransmissionLine:
    def __init__(self,Z0,eps_eff,Qi=np.inf) -> None:

        self.Z0 = Z0
        self.eps_eff = eps_eff
        self.Qi = Qi

    def wavelength(self, f):
        lmda = c0 / np.sqrt(self.eps_eff) / f
        return lmda

    def wavenumber(self, f):
        lmda = self.wavelength(f)
        k = 2 * np.pi / lmda * (1 - 1j / (2*self.Qi))
        return k

    def ABCD(self,f,l):
        g = 1j * self.wavenumber(f)
        g = g[:,np.newaxis,np.newaxis]

        A = np.cosh(g * l)
        B = self.Z0 * np.sinh(g * l)
        C = np.sinh(g * l) / self.Z0
        D = np.cosh(g * l)

        ABCD = np.block([[A,B],[C,D]])
        return ABCD


class Coupler:
    def __init__(self, f0, Ql, Z_termination, Qi=np.inf, topology='series', res_length='halfwave') -> None:
        self.f0 = f0
        self.Ql = Ql
        self.Qi = Qi
        if np.isinf(Qi):
            self.Qc = 2 * Ql
        else:
            self.Qc = 2 * Qi * Ql / (Qi - Ql)
        assert len(np.atleast_1d(Z_termination)) < 3, "Z_termination has too many components (max 2 components)"
        self.Z_termination = np.atleast_1d(Z_termination)

        assert topology in ('series','parallel')
        self.topology = topology
        
        assert res_length in ('halfwave','quarterwave')
        self.res_length = res_length
        
        self.C = self.capacitance()


    def capacitance(self,Qc=None):
        if not Qc:
            Qc = self.Qc
        
        R1, X1 = np.real(self.Z_termination[0]), np.imag(self.Z_termination[0])
        R2, X2 = np.real(self.Z_termination[-1]), np.imag(self.Z_termination[-1])
        
        n_cycles = {'halfwave':1,'quarterwave':2}.get(self.res_length)

        if self.topology == 'series':
            A = 1
            B = 2 * X1 + 2 * X2
            C = X1**2 + 2 * X1 * X2 + X2**2 - n_cycles * Qc / np.pi * 2 * R1 * R2 + (R1 + R2)**2
        elif self.topology == 'parallel':
            A = (R1 + R2)**2 + (X1 + X2)**2 - n_cycles * Qc / np.pi * 2 * R1 * R2
            B = 2 * (R1 * X2 + R2 * X1) * (R1 + R2) + 2 *(X1 * X2 - R1 * R2) * (X1 + X2)
            C = (R1 * X2 + R2 * X1)**2 + (X1 * X2 - R1 * R2)**2
        
        X = (-B - np.sqrt(B**2 - 4 * A * C)) / (2 * A)
    
        C_coup = -1 / (2 * np.pi * self.f0 * X)
        return C_coup


    def impedance(self,f):
        Z = -1j / (2 * np.pi * f * self.C)
        return Z
    

    def add_variance(self):
        pass

    def ABCD(self,f):
        Z = self.impedance(f)

        ABCD = abcd_seriesload(Z)

        return ABCD
        

class Resonator:
    def __init__(self, f0, Ql, TransmissionLine : TransmissionLine, Z_termination, sigma_f0=0, sigma_Qc=0) -> None:
        self.f0, self.Ql = (f0, Ql)
        _    , Ql_C1 = res_variance(f0,Ql,TransmissionLine.Qi,sigma_f0,sigma_Qc)
        _    , Ql_C2 = res_variance(f0,Ql,TransmissionLine.Qi,sigma_f0,sigma_Qc)
        f0_L1, _     = res_variance(f0,Ql,TransmissionLine.Qi,sigma_f0,sigma_Qc)
        
        self.TransmissionLine = TransmissionLine

        assert len(np.atleast_1d(Z_termination)) < 3, "Z_termination has too many components (max 2 components)"
        self.Z_termination = np.atleast_1d(Z_termination)

        self.Coupler1 = Coupler(f0=self.f0,Ql=Ql_C1,Z_termination=[TransmissionLine.Z0, self.Z_termination[0]],Qi=TransmissionLine.Qi)

        self.Coupler2 = Coupler(f0=self.f0,Ql=Ql_C2,Z_termination=[TransmissionLine.Z0, self.Z_termination[-1]],Qi=TransmissionLine.Qi)

        self.l_res = self.resonator_length(f0_L1)


    def resonator_length(self,f0_var):
        Z1 = self.Z_termination[0]
        Z2 = self.Z_termination[-1]
        Zres = self.TransmissionLine.Z0
        
        Z_Coupler1 = Coupler(f0=self.f0,Ql=self.Ql,Z_termination=[Zres, Z1],Qi=self.TransmissionLine.Qi).impedance(f0_var)
        Z_Coupler2 = Coupler(f0=self.f0,Ql=self.Ql,Z_termination=[Zres, Z2],Qi=self.TransmissionLine.Qi).impedance(f0_var)
        
        A = Z_Coupler2 + Z2
        
        kl = np.array(np.arctan( (Z1 - Z_Coupler1 - A) / (-1j * (Z1 * A / Zres - Z_Coupler1 * A / Zres - Zres)) ))
        kl[kl<0] = kl[kl<0] + np.pi

        lres = np.real(kl / self.TransmissionLine.wavenumber(f0_var))
        return lres

    def ABCD(self,f):
        ABCD = self.Coupler1.ABCD(f) @ self.TransmissionLine.ABCD(f,self.l_res) @ self.Coupler2.ABCD(f)
        
        return ABCD


class Reflector:
    def __init__(self, f0, Ql, TransmissionLine : TransmissionLine, Z_termination, sigma_f0=0, sigma_Qc=0) -> None:
        self.f0, self.Ql = (f0, Ql)
        _    , Ql_C1 = res_variance(f0,Ql,TransmissionLine.Qi,sigma_f0,sigma_Qc)
        f0_L1, _     = res_variance(f0,Ql,TransmissionLine.Qi,sigma_f0,sigma_Qc)

        self.TransmissionLine = TransmissionLine

        assert len(np.atleast_1d(Z_termination)) < 2, "Z_termination has too many components (max 1 component)"
        self.Z_termination = np.atleast_1d(Z_termination)

        self.Coupler = Coupler(f0=self.f0, Ql=Ql_C1, Z_termination=[TransmissionLine.Z0, self.Z_termination[0]], Qi=TransmissionLine.Qi, res_length='quarterwave')

        self.l_res = self.resonator_length(f0_L1)


    def resonator_length(self,f0_var):
        Z1 = self.Z_termination[0]
        Zres = self.TransmissionLine.Z0
        
        Z_Coupler = Coupler(f0=self.f0,Ql=self.Ql,Z_termination=[Zres, Z1],Qi=self.TransmissionLine.Qi).impedance(f0_var)
        
        kl = np.array(np.arctan( (Z1 - Z_Coupler) / (-1j * (Z1 / Zres - Z_Coupler  / Zres - Zres)) ))
        kl[kl<0] = kl[kl<0] + np.pi

        lres = np.real(kl / self.TransmissionLine.wavenumber(f0_var))
        return lres

    def ABCD(self,f):
        ABCD = abcd_shuntload(
            Zin_from_abcd(
                self.Coupler.ABCD(f) @ self.TransmissionLine.ABCD(f,self.l_res),
                0
            )
        )

        return ABCD