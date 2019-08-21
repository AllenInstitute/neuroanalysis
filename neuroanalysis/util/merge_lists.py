def merge_lists(a, b):
    """Return a list containing the unique items combined from a and b, in roughly
    the same order they appear in a and b.
    
    The input lists do _not_ need to be sorted, but any items that are common to both lists
    should appear in the same order.
    
    For example::
    
        >>> a = [1,2,5,7,8,9]
        >>> b = [2,3,4,5,8,10,11]
        >>> merge_lists(a, b)
        [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]    
    """
    a_inds = {k:i for i,k in enumerate(a)}
    c = a[:]
    i = len(a)
    for item in b[::-1]:
        if item in a_inds:
            i = a_inds[item]
        else:
            c.insert(i, item)
    return c
