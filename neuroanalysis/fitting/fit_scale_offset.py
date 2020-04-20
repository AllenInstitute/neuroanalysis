# coding: utf8

def fit_scale_offset(data, template):
    """Return the scale and offset needed to minimize the sum of squared errors between
    *data* and *template*::
    
        data ≈ scale * template + offset
    
    Credit: Clements & Bekkers 1997
    """
    assert data.shape == template.shape
    N = len(data)
    dsum = data.sum()
    tsum = template.sum()
    scale = ((template * data).sum() - tsum * dsum / N) / ((template**2).sum() - tsum**2 / N)
    offset = (dsum - scale * tsum) / N

    return scale, offset
