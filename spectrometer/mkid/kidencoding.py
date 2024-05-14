import numpy as np

def shuffle_kids(n_mkids,group_size):

    shuffled = np.empty(n_mkids,dtype="int")

    base = 0
    index = base
    for n in range(n_mkids):
        shuffled[index] = n
        index += group_size
        if index >= n_mkids:
            base += 1
            index = base
        
    return shuffled




def shuffle_kids2(n_mkids,base_repeat_matrix):
    base_repeat_matrix = np.array(base_repeat_matrix)
    if np.inner(base_repeat_matrix[:,0],base_repeat_matrix[:,1]) != n_mkids:
        raise ValueError("The inner product of the base_repeat matrix does not match the number of mkids")

    shuffled = np.array([])

    offset_index = 0
    flip = False
    for row in base_repeat_matrix:
        base_size = row[0]
        repeat = row[1]

        match base_size:
            case 5: 
                base = [1,3,5,2,4]
            case 6: 
                base = [1,4,2,5,3,6]
            case 7: 
                base = [1,4,7,2,5,3,6]
            case 8: 
                base = [1,4,7,2,5,8,3,6]
            case 9: 
                base = [1,4,7,2,5,8,3,6,9]
            case 10: 
                base = [1,4,7,10,2,5,8,3,6,9]
            case 12:
                # base = [1,4,7,10,2,5,8,11,3,6,9,12]
                base = [1,5,9,2,6,10,3,7,11,4,8,12]
            case _:
                base = [1,3,5,2,4]

        base_offset = np.add(base,offset_index)
    
        matrix = np.outer(np.arange(repeat),np.ones(len(base),dtype=int) * len(base))
        shuffled_part = np.add(matrix, base_offset)
        
        if flip:
            shuffled_part = np.fliplr(shuffled_part)
        
        print(shuffled_part)
        
        shuffled = np.append(shuffled, np.ravel(shuffled_part,order="F"))
        
        offset_index += (base_size * repeat)
        flip = not flip

    print(shuffled)
        
    return np.array(shuffled-1,dtype=int)
