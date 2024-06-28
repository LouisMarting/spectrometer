import numpy as np

def chain(*ABCDmatrices):
    """
    Function to facilitate chaining ABCD matrices.

    """
    assert len(ABCDmatrices) >= 2, "chain() needs at least two ABCD matrices"

    ABCD = ABCDmatrices[0]
    for ABCDmatrix in ABCDmatrices[1:]:
        ABCD = ABCD @ ABCDmatrix
    return ABCD

def unchain(*ABCDmatrices,at='front'):
    """
    Function to facilitate unchaining ABCD matrices

    
    Please take into account the non-commutativity of matrix multiplications, with at={'front','back'}.


    The order of matrices given is the order of unchaining. The first matrix is always the 'base'. 
    The 'at' argument defines from what side the matrices are unchained, in the order they were given. 
    The second matrix given is removed first, then the third, etc...

    TODO: Since we always use 2x2 matrices for the ABCD matrix calculations, 
    we can use the shorthand of the inverse for a 2x2 matrix to hopefully speed things up. edit: probs not, numpy is optimized already.

    """
    assert len(ABCDmatrices) >= 2, "unchain() needs at least two ABCD matrices"
    assert at in ('front','back')

    ABCD = ABCDmatrices[0]
    if at == 'front':
        for ABCDmatrix in ABCDmatrices[1:]:
            ABCD = np.linalg.inv(ABCDmatrix) @ ABCD
    elif at == 'back':
        for ABCDmatrix in ABCDmatrices[1:]:
            ABCD = ABCD @ np.linalg.inv(ABCDmatrix)
    
    return ABCD


def abcd_parallel(ABCD1,ABCD2):
    Y1 = abcd2y(ABCD1)
    Y2 = abcd2y(ABCD2)
    ABCD = y2abcd(Y1 + Y2)

    return ABCD


def abcd_shuntload(Z):
    """
    Calculate the ABCD matrix of a shunt load.
    """
    Z = np.array(Z,dtype=np.cfloat)  
    Z = Z[...,np.newaxis,np.newaxis]    # elegant solution to extend the dimension by two, 
                                        # allowing elegant use of np.ones_like(), np.zeros_like() and np.block()

    A = np.ones_like(Z)
    B = np.zeros_like(Z)
    C = 1 / Z
    D = np.ones_like(Z)

    ABCD = np.block([[A,B],[C,D]])

    return ABCD


def abcd_seriesload(Z):
    """
    Calculate the ABCD matrix of a series load.
    """
    Z = np.array(Z,dtype=np.cfloat)  
    Z = Z[...,np.newaxis,np.newaxis]    # elegant solution to extend the dimension by two, 
                                        # allowing elegant use of np.ones_like(), np.zeros_like() and np.block()
    
    A = np.ones_like(Z)
    B = Z
    C = np.zeros_like(Z)
    D = np.ones_like(Z)

    ABCD = np.block([[A,B],[C,D]])
    
    return ABCD


def Zin_from_abcd(ABCD,Z_L,load_pos='load'):
    assert load_pos in ('load','source')
    
    Z = abcd2z(ABCD)

    Z11 = Z[...,0,0]
    Z12 = Z[...,0,1]
    Z21 = Z[...,1,0]
    Z22 = Z[...,1,1]

    if load_pos == 'load':
        Z_in = Z11 - Z12 * Z21 / (Z22 + Z_L)
    elif load_pos == 'source':
        Z_in = Z22 - Z12 * Z21 / (Z11 + Z_L)

    return Z_in


def abcd2s(ABCD,Z0):
    """
    ABCD to S parameter transformation.
    
    See: "Conversions between S, Z, Y, H, ABCD, and T parameters which are valid for complex source and load impedances," D.A. Frickey, 1994.
    """
    S = np.empty_like(ABCD,dtype=np.cfloat)
    
    A = ABCD[...,0,0]
    B = ABCD[...,0,1]
    C = ABCD[...,1,0]
    D = ABCD[...,1,1]

    Z0_1 = np.atleast_1d(Z0)[0]
    Z0_2 = np.atleast_1d(Z0)[-1]

    den = A * Z0_2 + B + C * Z0_1 * Z0_2 + D * Z0_1
    
    S[...,0,0] = (A * Z0_2 + B - C * np.conj(Z0_1) * Z0_2 - D * np.conj(Z0_1)) / den
    S[...,0,1] = (2 * (A * D - B * C) * (np.real(Z0_1) * np.real(Z0_2)) **0.5) / den
    S[...,1,0] = (2 * (np.real(Z0_1) * np.real(Z0_2)) **0.5) / den
    S[...,1,1] = (-A * np.conj(Z0_2) + B - C * Z0_1 * np.conj(Z0_2) + D * Z0_1) / den

    return S


def abcd2z(ABCD):
    try:
        Z = np.empty_like(ABCD,dtype=np.cfloat)
        
        A = ABCD[...,0,0]
        B = ABCD[...,0,1]
        C = ABCD[...,1,0]
        D = ABCD[...,1,1]

        Z[...,0,0] = A / C
        Z[...,0,1] = (A * D - B * C) / C
        Z[...,1,0] = 1 / C
        Z[...,1,1] = D / C

        return Z
    except RuntimeWarning:
        raise RuntimeError


def abcd2y(ABCD):
    Y = np.empty_like(ABCD,dtype=np.cfloat)
    
    A = ABCD[...,0,0]
    B = ABCD[...,0,1]
    C = ABCD[...,1,0]
    D = ABCD[...,1,1]

    Y[...,0,0] = D / B
    Y[...,0,1] = (B * C - A * D) / B
    Y[...,1,0] = -1 / B
    Y[...,1,1] = A / B

    return Y


def s2abcd(S,Z0):
    ABCD = np.empty_like(S,dtype=np.cfloat)
    
    S11 = S[...,0,0]
    S12 = S[...,0,1]
    S21 = S[...,1,0]
    S22 = S[...,1,1]

    Z0_1 = np.atleast_1d(Z0)[0]
    Z0_2 = np.atleast_1d(Z0)[-1]

    den = 2*S21 * np.sqrt( np.real(Z0_1) * np.real(Z0_2) )
    
    ABCD[...,0,0] = (( np.conj(Z0_1) + S11 * Z0_1 ) / ( 1 - S22 ) + S12 * S21 * Z0_1 ) / den
    ABCD[...,0,1] = (( np.conj(Z0_1) + S11 * Z0_1 ) * ( np.conj(Z0_2) + S22 * Z0_2 ) - S12 * S21 * Z0_1 * Z0_2 ) / den
    ABCD[...,1,0] = (( 1 - S11 ) * ( 1 - S22 ) - S12 * S21 )/ den
    ABCD[...,1,1] = (( 1 - S11 ) * ( np.conj(Z0_2) + S22 * Z0_2 ) + S12 * S21 * Z0_2 ) / den

    return ABCD


def z2abcd():
    pass


def y2abcd(Y):
    ABCD = np.empty_like(Y,dtype=np.cfloat)

    Y11 = Y[...,0,0]
    Y12 = Y[...,0,1]
    Y21 = Y[...,1,0]
    Y22 = Y[...,1,1]

    ABCD[...,0,0] = -Y22 / Y21
    ABCD[...,0,1] = -1 / Y21
    ABCD[...,1,0] = (Y12 * Y21 - Y11 * Y22) / Y21
    ABCD[...,1,1] = -Y11 / Y21

    return ABCD

def t_circuit():
    pass

def pi_circuit():
    pass
