from pytest import raises
import numpy as np

from neuroanalysis.data import TSeries


def test_trace_timing():
    # Make sure sample timing is handled exactly--need to avoid fp error here
    a = np.random.normal(size=300)
    sr = 50000
    dt = 2e-5
    t = np.arange(len(a)) * dt
    
    # trace with no timing information 
    tr = TSeries(a)
    assert not tr.has_timing
    assert not tr.has_time_values
    with raises(TypeError):
        tr.dt
    with raises(TypeError):
        tr.sample_rate
    with raises(TypeError):
        tr.time_values
    with raises(TypeError):
        tr.time_at(0)
    with raises(TypeError):
        tr.index_at(0.1)
    with raises(TypeError):
        tr.value_at(0.1)
        
    view = tr[100:200]
    assert not tr.has_timing
    assert not tr.has_time_values

    # invalid data
    with raises(ValueError):
        TSeries(data=np.zeros((10, 10)))

    # invalid timing information
    with raises(TypeError):
        TSeries(data=a, dt=dt, time_values=t)
    with raises(TypeError):
        TSeries(data=a, sample_rate=sr, time_values=t)
    with raises(TypeError):
        TSeries(data=a, dt=dt, t0=0, time_values=t)
    with raises(TypeError):
        TSeries(data=a, dt=dt, t0=0, sample_rate=sr)
    with raises(ValueError):
        TSeries(data=a, time_values=t[:-1])

    # trace with only dt
    tr = TSeries(a, dt=dt)
    assert tr.dt == dt
    assert np.allclose(tr.sample_rate, sr)
    check_trace(tr, data=a, time_values=t, has_timing=True, has_time_values=False, regularly_sampled=True)

    # trace with only sample_rate
    tr = TSeries(a, sample_rate=sr)
    assert tr.dt == dt
    assert tr.sample_rate == sr
    assert np.all(tr.time_values == t)
    check_trace(tr, data=a, time_values=t, has_timing=True, has_time_values=False, regularly_sampled=True)
    
    # trace with only regularly-sampled time_values
    tr = TSeries(a, time_values=t)
    assert tr.dt == dt
    assert np.allclose(tr.sample_rate, sr)
    assert np.all(tr.time_values == t)
    check_trace(tr, data=a, time_values=t, has_timing=True, has_time_values=True, regularly_sampled=True)

    # trace with irregularly-sampled time values
    t1 = np.cumsum(np.random.normal(loc=1, scale=0.02, size=a.shape))
    tr = TSeries(a, time_values=t1)
    assert tr.dt == t1[1] - t1[0]
    assert np.all(tr.time_values == t1)
    check_trace(tr, data=a, time_values=t1, has_timing=True, has_time_values=True, regularly_sampled=False)


def check_trace(tr, data, time_values, has_timing, has_time_values, regularly_sampled):
    """Make sure trace timing is working for his trace and nested views on it.
    """
    check_timing(tr, data, time_values, has_timing, has_time_values, regularly_sampled)

    # modify t0
    t0 = time_values[0]
    tr.t0 = 123.4
    check_timing(tr, data=data, time_values=time_values+(123.4-t0), has_timing=has_timing, has_time_values=has_time_values, regularly_sampled=regularly_sampled)
    # and return back
    tr.t0 = t0

    if tr.has_time_values:  # fp errors for views are different based on whether time value array was specified
        time_values = (time_values + (123.4 - t0)) + (t0 - 123.4)
    check_timing(tr, data=data, time_values=time_values, has_timing=has_timing, has_time_values=has_time_values, regularly_sampled=regularly_sampled)

    # test view
    view = tr[100:200]
    t = tr.time_values
    assert view.t0 == t[100]
    assert view.time_values[0] == view.t0
    assert view._meta['dt'] == tr._meta['dt']
    assert view._meta['sample_rate'] == tr._meta['sample_rate']

    if tr.has_time_values:  # fp errors for views are different based on whether time value array was specified
        view_t = tr.time_values[100:200]
    else:
        view_t = tr.time_values[100] + tr.time_values[:100]
    check_timing(view, data=tr.data[100:200], time_values=view_t, has_timing=has_timing, has_time_values=has_time_values, regularly_sampled=regularly_sampled)

    # test nested view
    if tr.regularly_sampled:
        view2 = view.time_slice(view.t0 + 20*tr.dt, view.t0 + 50*tr.dt)
    
    else:
        # don't try time_slice with irregularly sampled
        view2 = view[20:50]
    assert view2.t0 == view.time_values[20] == tr.time_values[120]
    if tr.has_time_values:  # fp errors for views are different based on whether time value array was specified
        view_t2 = tr.time_values[120:150]
    else:
        view_t2 = view.time_values[20] + tr.time_values[:30]
    check_timing(view2, data=tr.data[120:150], time_values=view_t2, has_timing=has_timing, has_time_values=has_time_values, regularly_sampled=regularly_sampled)
    
    # modify view t0
    view.t0 = 0
    assert view.t0 == 0
    # make sure view has updated time values
    if tr.has_time_values:  # fp errors for views are different based on whether time value array was specified
        view_t = tr.time_values[100:200]-tr.time_values[100]
    else:
        view_t = tr.time_values[:100]
    check_timing(view, data=tr.data[100:200], time_values=view_t, has_timing=has_timing, has_time_values=has_time_values, regularly_sampled=regularly_sampled)
    # make sure the original trace and the sub-view are unaffected
    check_timing(tr, data, time_values, has_timing, has_time_values, regularly_sampled)
    check_timing(view2, data=tr.data[120:150], time_values=view_t2, has_timing=has_timing, has_time_values=has_time_values, regularly_sampled=regularly_sampled)


def check_timing(tr, data, time_values, has_timing, has_time_values, regularly_sampled):
    assert np.all(tr.time_values == time_values)
    assert tr.has_timing is has_timing
    assert tr.has_time_values is has_time_values
    assert tr.regularly_sampled is regularly_sampled

    assert tr.t_end == tr.time_values[-1]

    # test scalar argument for index_at, time_at, and value_at
    index = 5
    assert tr.value_at(time_values[index]) == tr.data[index]
    assert tr.time_at(index) == time_values[index]
    dt = time_values[index] - time_values[index-1]
    for t in [time_values[index], time_values[index]+dt*0.4, time_values[index]-dt*0.4]:
        index2 = tr.index_at(t)
        assert isinstance(index2, int)
        assert index2 == index

    # test array argument for index_at, time_at, and value_at
    t = [time_values[5], time_values[10]]
    indices = tr.index_at(t)
    assert np.all(indices == [5, 10])
    assert indices.dtype == int
    assert np.all(tr.value_at(t) == tr.data[indices])
    assert np.all(tr.time_at([5, 10]) == t)
    indices = np.arange(len(tr))
    assert np.all(tr.value_at(tr.time_at(indices)) == tr.data)
    assert np.all(tr.index_at(tr.time_at(indices)) == indices)

