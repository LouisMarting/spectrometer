import numpy as np
from scipy.signal import find_peaks

from ...utils.utils import ABCD_eye
from ...tline.transformations import abcd2s,Zin_from_abcd,abcd_shuntload,abcd_parallel,chain
from ...tline.components import TransmissionLine,Resonator,Reflector



class BaseFilter:
    def __init__(self, f0, Ql, TransmissionLines : dict) -> None:
        self.S_param = None
        self.f = None
        self.S11_absSq = None
        self.S21_absSq = None
        self.S31_absSq = None
        self.S41_absSq = None

        self.Ql_realized = None
        self.f0_realized = None

        self.f0 = f0
        self.Ql = Ql

        assert set(TransmissionLines.keys()) >= {'through','resonator','MKID'}, "TranmissionLines dict needs at least the keys: ('through','resonator','MKID')"
        self.TransmissionLines = TransmissionLines
        self.TransmissionLine_through : TransmissionLine = self.TransmissionLines['through']
        self.TransmissionLine_resonator : TransmissionLine = self.TransmissionLines['resonator']
        self.TransmissionLine_MKID : TransmissionLine = self.TransmissionLines['MKID']

        self.sep = self.TransmissionLine_through.wavelength(f0) / 4
    
    def ABCD_sep(self, f):
        ABCD = self.TransmissionLine_through.ABCD(f,self.sep)

        return ABCD
    
    
    def n_outputs(self):
        if isinstance(self,(ManifoldFilter,ReflectorFilter)):
            return 1
        elif isinstance(self,(DirectionalFilter,)):
            return 2
        else:
            return None

    def ABCD_shunt_termination(self, f, ABCD_to_termination):
        ABCD = abcd_shuntload(
            Zin_from_abcd(
                chain(
                    self.ABCD_sep(f),
                    ABCD_to_termination
                ),
                self.TransmissionLine_through.Z0
            )
        )

        return ABCD
    
    def ABCD(self, f):
        # In childs: Add code to construct filter
        pass

    def ABCD_to_MKID(self, f, ABCD_to_termination):
        ABCD_shunt_termination = self.ABCD_shunt_termination(f, ABCD_to_termination)
        # In childs: Add code to construct to MKID structure
        pass

    def S(self,f):
        if np.array_equal(self.f,f):
            return self.S_param
        else:
            s_parameter_array_size = self.n_outputs() + 2
            S = np.empty((len(f),s_parameter_array_size,),dtype=np.cfloat)
            S_through  = abcd2s(self.ABCD(f),self.TransmissionLine_through.Z0)
            S[:,0] = S_through[:,0,0] # S11
            S[:,1] = S_through[:,1,0] # S21

            for i,ABCD_to_one_output in enumerate(self.ABCD_to_MKID(f,ABCD_eye(len(f)))):
                S_one_output = abcd2s(ABCD_to_one_output,[self.TransmissionLine_through.Z0,self.TransmissionLine_MKID.Z0])

                index = i + 2
                S[:,index] = S_one_output[:,1,0] # Si1

            self.S_param = S
            self.f = f


            self.S11_absSq = np.abs(S[:,0])**2
            self.S21_absSq = np.abs(S[:,1])**2
            self.S31_absSq = np.abs(S[:,2])**2

            if self.n_outputs() == 2:
                self.S41_absSq = np.abs(self.S_param[:,3])**2

            return self.S_param
    
        
    def realized_parameters(self,n_interp=20):
        assert self.S_param is not None

        fq = np.linspace(self.f[0],self.f[-1],n_interp*len(self.f))
 
        S31_absSq_q = np.interp(fq,self.f,self.S31_absSq)

        i_peaks,peak_properties = find_peaks(S31_absSq_q,height=0.5*max(S31_absSq_q),prominence=0.02)

        i_peak = i_peaks[np.argmax(peak_properties["peak_heights"])]
        # f0, as realized in the filter (which is the peak with the highest height given a minimum relative height and prominence)
        self.f0_realized = fq[i_peak]

        # Find FWHM manually:
        HalfMaximum = S31_absSq_q[i_peak] / 2
        diff_from_HalfMaximum = np.abs(S31_absSq_q-HalfMaximum)

        # search window = +/- a number of filter widths
        search_range = [self.f0_realized-3*self.f0/self.Ql, self.f0_realized+3*self.f0/self.Ql]
        
        search_window = np.logical_and(fq > search_range[0],fq < self.f0_realized)
        i_HalfMaximum_lower = np.ma.masked_array(diff_from_HalfMaximum,mask=~search_window).argmin()

        search_window = np.logical_and(fq > self.f0_realized,fq < search_range[-1])
        i_HalfMaximum_higher = np.ma.masked_array(diff_from_HalfMaximum,mask=~search_window).argmin()

        fwhm = fq[i_HalfMaximum_higher] - fq[i_HalfMaximum_lower]

        self.Ql_realized = self.f0_realized / fwhm



        return self.f0_realized, self.Ql_realized


class ManifoldFilter(BaseFilter):
    def __init__(self, f0, Ql, TransmissionLines: dict, sigma_f0=0, sigma_Qc=0, compensate=True) -> None:
        super().__init__(f0, Ql, TransmissionLines)

        if compensate == True:
            self.Ql = Ql * 1.15
        else:
            self.Ql = Ql

        self.Resonator = Resonator(
            f0 = self.f0, 
            Ql = self.Ql, 
            TransmissionLine = self.TransmissionLine_resonator, 
            Z_termination = [self.TransmissionLine_through.Z0/2, self.TransmissionLine_MKID.Z0], 
            sigma_f0 = sigma_f0, 
            sigma_Qc = sigma_Qc
        )

    def ABCD(self, f):
        ABCD = abcd_shuntload(
            Zin_from_abcd(self.Resonator.ABCD(f),self.TransmissionLine_MKID.Z0)
        )
        
        return ABCD
    
    def ABCD_to_MKID(self, f, ABCD_to_termination):
        ABCD_shunt_termination = self.ABCD_shunt_termination(f, ABCD_to_termination)

        ABCD_to_MKID = chain(
            ABCD_shunt_termination,
            self.Resonator.ABCD(f)
        )
        
        return (ABCD_to_MKID,)


class ReflectorFilter(BaseFilter):
    def __init__(self, f0, Ql, TransmissionLines: dict, sigma_f0=0, sigma_Qc=0, compensate=True) -> None:
        super().__init__(f0, Ql, TransmissionLines)

        if compensate == True:
            self.Ql = Ql * 0.75
        else:
            self.Ql = Ql

        self.lmda_quarter = self.TransmissionLine_through.wavelength(f0) / 4
        # self.sep = self.lmda_quarter # quarter lambda is the standard BaseFilter separation

        # Impedance of resonator is equal to onesided connection, due to relfector creating an open condition
        self.Resonator = Resonator(
            f0 = self.f0, 
            Ql = self.Ql, 
            TransmissionLine = self.TransmissionLine_resonator, 
            Z_termination = [self.TransmissionLine_through.Z0, self.TransmissionLine_MKID.Z0], 
            sigma_f0 = sigma_f0, 
            sigma_Qc = sigma_Qc
        )

        self.Reflector = Reflector(
            f0 = self.f0, 
            Ql = self.Ql, 
            TransmissionLine = self.TransmissionLine_resonator, 
            Z_termination = self.TransmissionLine_through.Z0/2, 
            sigma_f0 = sigma_f0, 
            sigma_Qc = sigma_Qc
        )

    def ABCD(self, f):
        ABCD = chain(
            abcd_shuntload(Zin_from_abcd(self.Resonator.ABCD(f),self.TransmissionLine_MKID.Z0)),
            self.TransmissionLine_through.ABCD(f, l=self.lmda_quarter),
            self.Reflector.ABCD(f)
        )

        return ABCD

    def ABCD_to_MKID(self, f, ABCD_to_termination):
        ABCD_shunt_termination = self.ABCD_shunt_termination(f, ABCD_to_termination)

        ABCD_to_MKID = chain(
            ABCD_shunt_termination,
            self.Resonator.ABCD(f)
        )
        
        return (ABCD_to_MKID,)

    def ABCD_shunt_termination(self, f, ABCD_to_termination):
        ABCD = abcd_shuntload(
            Zin_from_abcd(
                chain(
                    self.TransmissionLine_through.ABCD(f, l=self.lmda_quarter),
                    self.Reflector.ABCD(f),
                    self.ABCD_sep(f),
                    ABCD_to_termination
                ),
                self.TransmissionLine_through.Z0
            )
        )

        return ABCD


class DirectionalFilter(BaseFilter):
    def __init__(self, f0, Ql, TransmissionLines : dict, sigma_f0=0, sigma_Qc=0, compensate=True) -> None:
        super().__init__(f0, Ql, TransmissionLines)

        if compensate == True:
            self.Ql = Ql * 0.95
        else:
            self.Ql = Ql

        self.lmda_quarter = self.TransmissionLine_through.wavelength(f0) / 4
        self.lmda_3quarter = self.TransmissionLine_MKID.wavelength(f0) * 3 / 4
        # self.sep = self.lmda_quarter # quarter lambda is the standard BaseFilter separation

        self.Resonator1 = Resonator(
            f0 = self.f0, 
            Ql = self.Ql, 
            TransmissionLine = self.TransmissionLine_resonator, 
            Z_termination = [self.TransmissionLine_through.Z0/2, self.TransmissionLine_MKID.Z0/2], 
            sigma_f0 = sigma_f0, 
            sigma_Qc = sigma_Qc
        )

        self.Resonator2 = Resonator(
            f0 = self.f0, 
            Ql = self.Ql, 
            TransmissionLine = self.TransmissionLine_resonator, 
            Z_termination = [self.TransmissionLine_through.Z0/2, self.TransmissionLine_MKID.Z0/2], 
            sigma_f0 = sigma_f0, 
            sigma_Qc = sigma_Qc
        )
        
    
    def ABCD(self, f):
        ABCD_lower = chain(
            self.Resonator1.ABCD(f), 
            abcd_shuntload(self.TransmissionLine_MKID.Z0),
            self.TransmissionLine_MKID.ABCD(f,self.lmda_3quarter),
            abcd_shuntload(self.TransmissionLine_MKID.Z0),
            self.Resonator2.ABCD(f)
        )
        
        ABCD = abcd_parallel(ABCD_lower,self.TransmissionLine_through.ABCD(f, l=self.lmda_quarter))

        return ABCD


    def ABCD_to_MKID(self, f, ABCD_to_termination):
        ABCD_shunt_termination = self.ABCD_shunt_termination(f, ABCD_to_termination)

        ABCD_upper = chain(
            self.TransmissionLine_through.ABCD(f, l=self.lmda_quarter),
            ABCD_shunt_termination,
            self.Resonator2.ABCD(f)
        )

        ABCD_lower = chain(
            self.Resonator1.ABCD(f),
            abcd_shuntload(self.TransmissionLine_MKID.Z0),
            self.TransmissionLine_MKID.ABCD(f,self.lmda_3quarter)
        )

        ABCD_port4 = abcd_parallel(ABCD_upper,ABCD_lower)

        ABCD_upper_port3 = chain(
            ABCD_upper,
            abcd_shuntload(self.TransmissionLine_MKID.Z0),
            self.TransmissionLine_MKID.ABCD(f,self.lmda_3quarter)
        )
        
        ABCD_port3 = abcd_parallel(ABCD_upper_port3,self.Resonator1.ABCD(f))

        return (ABCD_port3,ABCD_port4,)

