import numpy as np
from scipy.signal import find_peaks

import copy
from functools import lru_cache

from ...utils.utils import ABCD_eye
from ...tline.transformations import abcd2s,chain,unchain
from .filters import BaseFilter, DirectionalFilter, ManifoldFilter, ReflectorFilter, DeletedFilter


# Ways to define a filterbank
# give 2 out of 3 of R, Ql, oversampling and f0_min, f0_max. Calculate list of f0
# give list of f0, and Ql/oversampling


class Filterbank:
    def __init__(self, FilterClass : BaseFilter, TransmissionLines : dict, f0, Ql) -> None:
        self.FilterClass = FilterClass
        self.TransmissionLines = TransmissionLines # to remove
        self.f0 = f0
        self.Ql = Ql # assume now same value for all filters, but could vary it later ?? maybe not in init, part of variance functions

        # Assume regular spacing of f0 for R calculation
        df0 = np.abs(f0[1] - f0[0])
        self.R = np.round(f0[0] / df0,decimals=3)

        self.oversampling = self.R / Ql
        

        self.Filters = np.empty(len(f0),dtype=BaseFilter)
        for i,fi in enumerate(f0):
            self.Filters[i] = FilterClass(f0=fi, Ql=Ql, TransmissionLines = copy.deepcopy(TransmissionLines),compensate=False)


    # this should be moved to the filterbank submodule as a separate function
    @staticmethod
    def f0_range(f0_min, f0_max, R):
        '''
        Returns a list of f0 based on a frequency range and the desired spectral resolution. 
        The list is ordered from highest frequency to lowest frequency.
        And the list is aligned to the lowest frequency value.
        '''

        #todo change from floor to ceil. to cover full band, instead of coming short sometimes.
        n_filters = int(np.floor(1 + np.log10(f0_max / f0_min) / np.log10(1 + 1 / R))) 
        
        f0 = np.zeros(n_filters)
        f0[0] = f0_min
        for i in np.arange(1,n_filters):
            f0[i] = f0[i-1] + f0[i-1] / R
        return np.flip(f0)

    @classmethod
    def from_parameters(cls, FilterClass : BaseFilter, TransmissionLines : dict, f0_min, f0_max, R=None, Ql=None, oversampling=None):
        '''
        Provide an alternative way of generating a filterbank.
        '''
        assert sum(p is None for p in (R, Ql, oversampling)) == 1, "Exactly one parameter of R, Ql, oversampling should not be defined, \
                                                                    as it is inferred from the other two"

        if oversampling is None:
            oversampling = R / Ql
        elif Ql is None:
            Ql = R / oversampling
        elif R is None:
            R = oversampling * Ql
        
        f0 = cls.f0_range(f0_min, f0_max, R)

        return cls(FilterClass, TransmissionLines, f0, Ql)


    # @lru_cache(maxsize=2)
    def S(self,f):
    
        Z0_thru = self.TransmissionLines['through'].Z0
        Z0_mkid = self.TransmissionLines['MKID'].Z0
        
        # get full 4D matrixes of the preceding and succeeding part of the filterbank at each filter for all freq.
        ABCD_preceding, ABCD_succeeding, ABCD_total = self._generate_preceding_succeeding_total(f)
        
        # define empty S parameter array
        i_existing_filters = [i for i,Filter in enumerate(self.Filters) if isinstance(Filter, (ManifoldFilter,DirectionalFilter,ReflectorFilter))]
        existing_filters = self.Filters[i_existing_filters]

        s_array_size = self.FilterClass.n_outputs * len(existing_filters) + 2
        S = np.empty((len(f),s_array_size,),dtype=np.cdouble)

        for i,Filter in enumerate(existing_filters):
            Filter : BaseFilter # set the expected datatype of Filter

            # Calculate the equivalent ABCD to the ith detector
            ABCD_to_MKID = Filter.ABCD_to_MKID(f,chain(Filter.ABCD_sep(f),ABCD_succeeding[i_existing_filters[i],:,:,:]))

            for j,ABCD_to_one_output in enumerate(ABCD_to_MKID):
                ABCD_through_filter = chain(
                    ABCD_preceding[i_existing_filters[i],:,:,:],
                    ABCD_to_one_output
                )
                S_one_output = abcd2s(ABCD_through_filter,[Z0_thru,Z0_mkid])

                index = (self.FilterClass.n_outputs * i)+j+2
                
                S[:,index] = S_one_output[:,1,0] # Si1
            
        
        S_full_FB = abcd2s(ABCD_total,Z0_thru)
        S[:,0] = S_full_FB[:,0,0] # S11
        S[:,1] = S_full_FB[:,1,0] # S21

        self.ABCD = ABCD_total
        self.S_param = S
        self.f = f

        self.S11_absSq = np.abs(S[:,0])**2
        self.S21_absSq = np.abs(S[:,1])**2
        if self.FilterClass.n_outputs == 2:
            self.S31_absSq_list = np.abs(S[:,2::2])**2 + np.abs(S[:,3::2])**2
        else:
            self.S31_absSq_list = np.abs(S[:,2:])**2

        return self.S_param
    

    def _generate_preceding_succeeding_total(self,f):
        # Generate the arrays for storing the preceding and succeeding arrays in the FB calculation
        # Hopefully, pregenerating helps with improving the code.

        size = (len(self.f0),len(f))
        ABCD_preceding = ABCD_eye(size)
        ABCD_succeeding = ABCD_eye(size)
        
        ABCD_preceding_temp = ABCD_eye(len(f))
        # Calculate a full filterbank chain
        for i, Filter in enumerate(self.Filters):
            Filter : BaseFilter # set the expected datatype of Filter
            ABCD_preceding[i,:,:,:] = ABCD_preceding_temp

            # chain all the preceding together for making the total later
            ABCD_preceding_temp = chain(
                ABCD_preceding_temp,
                Filter.ABCD(f),
                Filter.ABCD_sep(f),
            )

        

        ABCD_total = ABCD_preceding_temp
        ABCD_succeeding = unchain(
            ABCD_total,
            ABCD_preceding
        )
        ABCD_succeeding[:-1,:,:,:] = ABCD_succeeding[1:,:,:,:]
        ABCD_succeeding[-1,:,:,:] = ABCD_eye(len(f))

        return ABCD_preceding, ABCD_succeeding, ABCD_total


    def delete_filters(self,filter_indices):
        for i in filter_indices:
            Filter_del : BaseFilter = self.Filters[i]
            thru_line_electrical_length = 1/4
            if type(Filter_del) == ManifoldFilter:
                thru_line_electrical_length = 0

            self.Filters[i] = DeletedFilter(Filter_del.f0,Filter_del.Ql,Filter_del.TransmissionLines,thru_line_electrical_length)


    # # This should combine with the function above
    # def S_sparse(self,f,indices):
    #     sparse_indices = set(indices)

    #     Z0_thru = self.TransmissionLines['through'].Z0
    #     Z0_mkid = self.TransmissionLines['MKID'].Z0
        
    #     ABCD_preceding = ABCD_eye(len(f))
    #     ABCD_succeeding = ABCD_eye(len(f))

    #     ABCD_list = ABCD_eye((len(f),len(self.f0)))
    #     ABCD_sep_list = ABCD_eye((len(f),len(self.f0)))
        

    #     # Calculate a full filterbank chain
    #     for i, Filter in enumerate(self.Filters):
    #         Filter : BaseFilter # set the expected datatype of Filter
            

    #         # Eventually, these indexed lists could be replaced by cached versions.
    #         ABCD_list[:,i,:,:] = Filter.ABCD(f)
    #         ABCD_sep_list[:,i,:,:] = Filter.ABCD_sep(f)

    #         # Can we use np.insert() for these and do this calc fastecr outside of this for loop?
    #         if i in sparse_indices:
    #             ABCD_succeeding = chain(
    #                 ABCD_succeeding,
    #                 ABCD_list[:,i,:,:],
    #                 ABCD_sep_list[:,i,:,:],
    #             )
    #         else:
    #             ABCD_succeeding = chain(
    #                 ABCD_succeeding,
    #                 ABCD_sep_list[:,i,:,:],
    #                 ABCD_sep_list[:,i,:,:],
    #             )
        
    #     s_parameter_array_size = Filter.n_outputs() * len(sparse_indices) + 2
    #     S = np.empty((len(f),s_parameter_array_size,),dtype=np.cdouble)

    #     for i,Filter in enumerate(self.Filters):
    #         Filter : BaseFilter # set the expected datatype of Filter
            
    #         # Remove the ith filter from the succeeding filters
    #         if i in sparse_indices:
    #             ABCD_succeeding = unchain(
    #                 ABCD_succeeding,
    #                 ABCD_list[:,i,:,:],
    #                 ABCD_sep_list[:,i,:,:]
    #             )
    #         else:
    #             ABCD_succeeding = unchain(
    #                 ABCD_succeeding,
    #                 ABCD_sep_list[:,i,:,:],
    #                 ABCD_sep_list[:,i,:,:]
    #             )

    #         if i in sparse_indices:
    #             # Calculate the equivalent ABCD to the ith detector
    #             ABCD_to_MKID = Filter.ABCD_to_MKID(f,ABCD_succeeding)

    #             assert len(ABCD_to_MKID) == Filter.n_outputs(), "Something seriously wrong here"

    #             for j,ABCD_to_one_output in enumerate(ABCD_to_MKID):
    #                 ABCD_through_filter = chain(
    #                     ABCD_preceding,
    #                     ABCD_to_one_output
    #                 )
    #                 S_one_output = abcd2s(ABCD_through_filter,[Z0_thru,Z0_mkid])

    #                 index = (Filter.n_outputs() * int(np.argwhere(np.array(list(sparse_indices)) == i)) )+j+2
                    
    #                 S[:,index] = S_one_output[:,1,0] # Si1
            
    #         if i in sparse_indices:
    #             ABCD_preceding = chain(
    #                 ABCD_preceding,
    #                 ABCD_list[:,i,:,:],
    #                 ABCD_sep_list[:,i,:,:]
    #             )
    #         else:
    #             ABCD_preceding = chain(
    #                 ABCD_preceding,
    #                 ABCD_sep_list[:,i,:,:],
    #                 ABCD_sep_list[:,i,:,:]
    #             )
        
    #     S_full_FB = abcd2s(ABCD_preceding,Z0_thru)
    #     S[:,0] = S_full_FB[:,0,0] # S11
    #     S[:,1] = S_full_FB[:,1,0] # S21

    #     self.S_param = S
    #     self.f = f

    #     self.S11_absSq = np.abs(S[:,0])**2
    #     self.S21_absSq = np.abs(S[:,1])**2
    #     if self.Filters[0].n_outputs() == 2:
    #         self.S31_absSq_list = np.abs(S[:,2::2])**2 + np.abs(S[:,3::2])**2
    #     else:
    #         self.S31_absSq_list = np.abs(S[:,2:])**2

    #     return self.S_param

    # def realized_parameters(self,n_interp=20):
    #     assert self.S_param is not None

    #     fq = np.linspace(self.f[0],self.f[-1],n_interp*len(self.f))
    #     dfq = fq[1] - fq[0]
    #     self.f0_realized = np.zeros(len(self.f0))
    #     self.Ql_realized = np.zeros(len(self.f0))
    #     self.inband_filter_eff = np.zeros(len(self.f0))
    #     self.inband_fraction = np.zeros(len(self.f0))

    #     for i in np.arange(len(self.f0)):
    #         if n_interp > 1:
    #             S31_absSq_q = np.interp(fq,self.f,self.S31_absSq_list[:,i])
    #         else:
    #             S31_absSq_q = self.S31_absSq_list[:,i]

    #         n_tries = 5
    #         width_in_samples = int(np.ceil(self.f0[i] / self.Ql / dfq))
    #         max_response = np.max(S31_absSq_q)
    #         for count,prom in enumerate(np.linspace(0.4,0.1,n_tries)):
    #             try:
    #                 # prominence is always tricky, add try except loop to gradually decrease prominence in case of no peaks
    #                 # i_peaks,_ = find_peaks(S31_absSq_q,height=0.5*max_response,distance=width_in_samples,prominence=prom*max_response,wlen=10*width_in_samples)
    #                 i_peaks,_ = find_peaks(S31_absSq_q,height=0.5*max_response,distance=width_in_samples)

    #                 i_peak = i_peaks[np.argmin(np.abs(fq[i_peaks]-self.f0[i]))]
    #             except ValueError:
    #                 if count >= (n_tries-1):
    #                     # plt.plot(S31_absSq_q)
    #                     # plt.plot(i_peaks,S31_absSq_q[i_peaks],"x")
    #                     # plt.show()
    #                     # print(i)
    #                     # plt.plot(self.S31_absSq_list[i])
    #                     raise Exception(f'Fitting peaks could not find a filter response peak with prom = {prom}')
    #             else:
    #                 break
                

            
    #         # f0, as realized in the filterbank (which is the peak with the highest height given a minimum relative height and prominence)
    #         self.f0_realized[i] = fq[i_peak]

    #         # Find FWHM manually:
    #         HalfMaximum = S31_absSq_q[i_peak] / 2
    #         diff_from_HalfMaximum = np.abs(S31_absSq_q-HalfMaximum)

    #         # search window = +/- a number of filter widths
    #         search_range = [self.f0_realized[i]-3*self.f0[i]/self.Ql, self.f0_realized[i]+3*self.f0[i]/self.Ql]
            
    #         search_window = np.logical_and(fq > search_range[0],fq < self.f0_realized[i])
    #         i_HalfMaximum_lower = np.ma.masked_array(diff_from_HalfMaximum,mask=~search_window).argmin()

    #         search_window = np.logical_and(fq > self.f0_realized[i],fq < search_range[-1])
    #         i_HalfMaximum_higher = np.ma.masked_array(diff_from_HalfMaximum,mask=~search_window).argmin()

    #         fwhm = fq[i_HalfMaximum_higher] - fq[i_HalfMaximum_lower]

    #         self.Ql_realized[i] = self.f0_realized[i] / fwhm

    #         # inband_filter_eff
    #         # inband_fraction
    #         i_f_max_fb = np.argmin(np.abs(fq-self.f0_realized[0]*1.01))


    #         self.inband_filter_eff[i] = np.sum(S31_absSq_q[i_HalfMaximum_lower:i_HalfMaximum_higher+1]) / (i_HalfMaximum_higher+1-i_HalfMaximum_lower)
    #         self.inband_fraction[i] = np.sum(S31_absSq_q[i_HalfMaximum_lower:i_HalfMaximum_higher+1]) / np.sum(S31_absSq_q[:i_f_max_fb])

    #     return self.f0_realized, self.Ql_realized, self.inband_filter_eff, self.inband_fraction

    # def realized_parameters_sparse(self,indices,n_interp=20):
    #     assert self.S_param is not None

    #     fq = np.linspace(self.f[0],self.f[-1],n_interp*len(self.f))
    #     dfq = fq[1] - fq[0]
    #     self.f0_realized = np.zeros_like(indices)
    #     self.Ql_realized = np.zeros_like(indices)
    #     self.inband_filter_eff = np.zeros_like(indices)
    #     self.inband_fraction = np.zeros_like(indices)

    #     for i,sparse_index in enumerate(indices):
    #         if n_interp > 1:
    #             S31_absSq_q = np.interp(fq,self.f,self.S31_absSq_list[:,i])
    #         else:
    #             S31_absSq_q = self.S31_absSq_list[:,i]

    #         n_tries = 5
    #         width_in_samples = int(np.ceil(self.f0[i] / self.Ql / dfq))
    #         max_response = np.max(S31_absSq_q)
    #         for count,prom in enumerate(np.linspace(0.4,0.1,n_tries)):
    #             try:
    #                 # prominence is always tricky, add try except loop to gradually decrease prominence in case of no peaks
    #                 # i_peaks,_ = find_peaks(S31_absSq_q,height=0.5*max_response,distance=width_in_samples,prominence=prom*max_response,wlen=10*width_in_samples)
    #                 i_peaks,_ = find_peaks(S31_absSq_q,height=0.5*max_response,distance=width_in_samples)

    #                 i_peak = i_peaks[np.argmin(np.abs(fq[i_peaks]-self.f0[i]))]
    #             except ValueError:
    #                 if count >= (n_tries-1):
    #                     # plt.plot(S31_absSq_q)
    #                     # plt.plot(i_peaks,S31_absSq_q[i_peaks],"x")
    #                     # plt.show()
    #                     # print(i)
    #                     # plt.plot(self.S31_absSq_list[i])
    #                     raise Exception(f'Fitting peaks could not find a filter response peak with prom = {prom}')
    #             else:
    #                 break
                

            
    #         # f0, as realized in the filterbank (which is the peak with the highest height given a minimum relative height and prominence)
    #         self.f0_realized[i] = fq[i_peak]

    #         # Find FWHM manually:
    #         HalfMaximum = S31_absSq_q[i_peak] / 2
    #         diff_from_HalfMaximum = np.abs(S31_absSq_q-HalfMaximum)

    #         # search window = +/- a number of filter widths
    #         search_range = [self.f0_realized[i]-3*self.f0[sparse_index]/self.Ql, self.f0_realized[i]+3*self.f0[sparse_index]/self.Ql]
            
    #         search_window = np.logical_and(fq > search_range[0],fq < self.f0_realized[i])
    #         i_HalfMaximum_lower = np.ma.masked_array(diff_from_HalfMaximum,mask=~search_window).argmin()

    #         search_window = np.logical_and(fq > self.f0_realized[i],fq < search_range[-1])
    #         i_HalfMaximum_higher = np.ma.masked_array(diff_from_HalfMaximum,mask=~search_window).argmin()

    #         fwhm = fq[i_HalfMaximum_higher] - fq[i_HalfMaximum_lower]

    #         self.Ql_realized[i] = self.f0_realized[i] / fwhm

    #         # inband_filter_eff
    #         # inband_fraction
    #         i_f_max_fb = np.argmin(np.abs(fq-self.f0_realized[0]*1.01))


    #         self.inband_filter_eff[i] = np.sum(S31_absSq_q[i_HalfMaximum_lower:i_HalfMaximum_higher+1]) / (i_HalfMaximum_higher+1-i_HalfMaximum_lower)
    #         self.inband_fraction[i] = np.sum(S31_absSq_q[i_HalfMaximum_lower:i_HalfMaximum_higher+1]) / np.sum(S31_absSq_q[:i_f_max_fb])

    #     return self.f0_realized, self.Ql_realized, self.inband_filter_eff, self.inband_fraction
    
    # def reset_and_shuffle(self):
    #     for i in np.arange(self.n_filters):
    #         self.Filters[i] = self.FilterClass(f0=self.f0[i], Ql=self.Ql, TransmissionLines = self.TransmissionLines, sigma_f0=self.sigma_f0, sigma_Qc=self.sigma_Qc)

    #     self.S_param = None
    #     self.f = None
    #     self.S11_absSq = None
    #     self.S21_absSq = None
    #     self.S31_absSq_list = None

    #     self.f0_realized = None
    #     self.Ql_realized = None


#### REPLACE WITH Delete filters
    # def sparse_filterbank(self,indices):
    #     self.S_param = None
    #     self.f = None
    #     self.S11_absSq = None
    #     self.S21_absSq = None
    #     self.S31_absSq_list = None

    #     self.f0_realized = None
    #     self.Ql_realized = None
    #     self.inband_filter_eff = None
    #     self.inband_fraction = None

    #     self.n_filters = len(indices)

    #     #select only the sparse filters.
    #     f0_full = self.f0
    #     self.f0 = f0_full[indices]

    #     self.Filters = np.empty(self.n_filters,dtype=BaseFilter)
    #     for i in np.arange(self.n_filters):
    #         self.Filters[i] = self.FilterClass(f0=self.f0[i], Ql=self.Ql, TransmissionLines = copy.deepcopy(self.TransmissionLines), sigma_f0=self.sigma_f0, sigma_Qc=self.sigma_Qc, compensate=self.compensate)

    def coupler_variance(self,Qc_shifted):

        for Filter in self.Filters:
            Filter : BaseFilter
            if type(Filter) == DeletedFilter:
                continue

            if type(Filter) == DirectionalFilter:
                Filter.Resonator1.Coupler1.C = Filter.Resonator1.Coupler1.capacitance(Qc_shifted)
                Filter.Resonator1.Coupler2.C = Filter.Resonator1.Coupler2.capacitance(Qc_shifted)
                Filter.Resonator2.Coupler1.C = Filter.Resonator2.Coupler1.capacitance(Qc_shifted)
                Filter.Resonator2.Coupler2.C = Filter.Resonator2.Coupler2.capacitance(Qc_shifted)
            elif type(Filter) == ReflectorFilter:
                Filter.Resonator.Coupler1.C = Filter.Resonator.Coupler1.capacitance(Qc_shifted)
                Filter.Resonator.Coupler2.C = Filter.Resonator.Coupler2.capacitance(Qc_shifted)
                Filter.Reflector.Coupler.C = Filter.Reflector.Coupler.capacitance(Qc_shifted)
            elif type(Filter) == ManifoldFilter:
                Filter.Resonator.Coupler1.C = Filter.Resonator.Coupler1.capacitance(Qc_shifted)
                Filter.Resonator.Coupler2.C = Filter.Resonator.Coupler2.capacitance(Qc_shifted)
        return
    
    
    def eps_eff_shift(self,relative_shift):

        for Filter in self.Filters:
            Filter : BaseFilter
            if type(Filter) == DeletedFilter:
                continue

            if type(Filter) == DirectionalFilter:
                Filter.TransmissionLine_through.eps_eff = Filter.TransmissionLine_through.eps_eff * relative_shift
                Filter.TransmissionLine_MKID.eps_eff = Filter.TransmissionLine_MKID.eps_eff * relative_shift
                Filter.Resonator1.TransmissionLine.eps_eff = Filter.Resonator1.TransmissionLine.eps_eff * relative_shift
                Filter.Resonator2.TransmissionLine.eps_eff = Filter.Resonator2.TransmissionLine.eps_eff * relative_shift
                # Filter.sep = Filter.TransmissionLine_through.wavelength(Filter.f0) / 4
            elif type(Filter) == ReflectorFilter:
                Filter.TransmissionLine_through.eps_eff = Filter.TransmissionLine_through.eps_eff * relative_shift
                Filter.TransmissionLine_MKID.eps_eff = Filter.TransmissionLine_MKID.eps_eff * relative_shift
                Filter.Resonator.TransmissionLine.eps_eff = Filter.Resonator.TransmissionLine.eps_eff * relative_shift
                Filter.Reflector.TransmissionLine.eps_eff = Filter.Reflector.TransmissionLine.eps_eff * relative_shift
                # Filter.sep = Filter.TransmissionLine_through.wavelength(Filter.f0) / 4
                # Filter.lmda_quarter = Filter.TransmissionLine_through.wavelength(Filter.f0) / 4
            elif type(Filter) == ManifoldFilter:
                Filter.TransmissionLine_through.eps_eff = Filter.TransmissionLine_through.eps_eff * relative_shift
                Filter.TransmissionLine_MKID.eps_eff = Filter.TransmissionLine_MKID.eps_eff * relative_shift
                Filter.Resonator.TransmissionLine.eps_eff = Filter.Resonator.TransmissionLine.eps_eff * relative_shift
                # Filter.sep = Filter.TransmissionLine_through.wavelength(Filter.f0) / 4
                # Filter.lmda_quarter = Filter.TransmissionLine_through.wavelength(Filter.f0) / 4
                # Filter.lmda_3quarter = Filter.TransmissionLine_MKID.wavelength(Filter.f0) * 3 / 4
        return

