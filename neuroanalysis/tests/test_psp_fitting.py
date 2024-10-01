import glob
import os

import pytest

import neuroanalysis.data
from neuroanalysis.fitting.psp import PspFitTestCase

path = os.path.join(os.path.dirname(neuroanalysis.__file__), '..', 'test_data', 'evoked_synaptic_events', '*.pkl')
psp_files = sorted(glob.glob(path))

test_ui = None


@pytest.mark.parametrize('test_file', psp_files)
def test_psp_fitting(request, test_file):
    from neuroanalysis.ui.psp_fitting import PspFitTestUi

    global test_ui
    audit = request.config.getoption('audit')
    if audit and test_ui is None:
        test_ui = PspFitTestUi()

    print("test:", test_file)
    tc = PspFitTestCase()
    tc.load_file(test_file)
    if audit:
        tc.audit_test(test_ui)
    else:
        tc.run_test()









# test_data_dir = os.path.join(os.path.dirname(neuroanalysis.__file__), '..', 'test_data', 'test_psp_fit')                                                                                                                                      

    
# def test_psp_fitting():
#     """Test psp_fit function against data from test directory.  Note that this 
#     test is highly sensitive.  If this test fails check_psp_fitting can be 
#     used to investigate whether the differences are substantial. Many things
#     can change the output of the fit slightly that would not be considered a real
#     difference from a scientific perspective.  i.e. numbers off by a precision 
#     of e-6.  One should look though the plots created by check_psp_fitting if there
#     is any question.  Unexpected things such as the order of the parameters fed 
#     to the function can create completely different fits.
#     """
#     plotting=True # specifies whether to make plots of fitting results
        
#     test_data_files=[os.path.join(test_data_dir,f) for f in os.listdir(test_data_dir)] #list of test files
#     for file in sorted(test_data_files):
# #    for file in ['test_psp_fit/1492546902.92_2_6stacked.json']: order of parameters affects this fit
#         print('file', file)
#         test_dict=json.load(open(file)) # load test data
#         avg_trace=neuroanalysis.data.TSeries(data=np.array(test_dict['input']['data']), dt=test_dict['input']['dt']) # create TSeries object
#         psp_fits = fit_psp(avg_trace, 
#                            xoffset=(14e-3, -float('inf'), float('inf')),
#                            weight=np.array(test_dict['input']['weight']),
#                            sign=test_dict['input']['amp_sign'], 
#                            stacked=test_dict['input']['stacked'] 
#                             )                        
        
#         assert test_dict['out']['best_values']==psp_fits.best_values, \
#             "Best values don't match. Run check_psp_fitting for more information"

#         assert test_dict['out']['best_fit']==psp_fits.best_fit.tolist(), \
#             "Best fit traces don't match. Run check_psp_fitting for more information"

#         assert test_dict['out']['nrmse']==float(psp_fits.nrmse()), \
#            "Nrmse doesn't match. Run check_psp_fitting for more information"

 

# def check_psp_fitting():
#     """Plots the results of the current fitting with the save fits and denotes 
#     when there is a change.   
#     """
#     plotting=True # specifies whether to make plots of fitting results
        
#     test_data_files=[os.path.join(test_data_dir,f) for f in os.listdir(test_data_dir)] #list of test files
#     for file in sorted(test_data_files):
# #    for file in ['test_psp_fit/1492546902.92_2_6stacked.json']: order of parameters affects this fit
#         print('file', file)
#         test_dict=json.load(open(file)) # load test data
#         avg_trace=neuroanalysis.data.TSeries(data=np.array(test_dict['input']['data']), dt=test_dict['input']['dt']) # create TSeries object
#         psp_fits = fit_psp(avg_trace, 
#                            weight=np.array(test_dict['input']['weight']),
#                            xoffset=(14e-3, -float('inf'), float('inf')),
#                            sign=test_dict['input']['amp_sign'], 
#                            stacked=test_dict['input']['stacked'] 
#                             )                        
        
#         change_flag=False
#         if test_dict['out']['best_values']!=psp_fits.best_values:     
#             print('  the best values dont match')
#             print('\tsaved', test_dict['out']['best_values'])
#             print('\tobtained', psp_fits.best_values)
#             change_flag=True
            
#         if test_dict['out']['best_fit']!=psp_fits.best_fit.tolist():
#             print('  the best fit traces dont match')
#             print('\tsaved', test_dict['out']['best_fit'])
#             print('\tobtained', psp_fits.best_fit.tolist())
#             change_flag=True
        
#         if test_dict['out']['nrmse']!=float(psp_fits.nrmse()):
#             print('  the nrmse doesnt match')
#             print('\tsaved', test_dict['out']['nrmse'])
#             print('\tobtained', float(psp_fits.nrmse()))
#             change_flag=True
            
#         if plotting:
#             import matplotlib.pylab as mplt
#             fig=mplt.figure(figsize=(20,8))
#             ax=fig.add_subplot(1,1,1)
#             ax2=ax.twinx()
#             ax.plot(avg_trace.time_values, psp_fits.data*1.e3, 'b', label='data')
#             ax.plot(avg_trace.time_values, psp_fits.best_fit*1.e3, 'g', lw=5, label='current best fit')
#             ax2.plot(avg_trace.time_values, test_dict['input']['weight'], 'r', label='weighting')
#             if change_flag is True:
#                 ax.plot(avg_trace.time_values, np.array(test_dict['out']['best_fit'])*1.e3, 'k--', lw=5, label='original best fit')
#                 mplt.annotate('CHANGE', xy=(.5, .5), xycoords='figure fraction', fontsize=40)
#             ax.legend()
#             mplt.title(file + ', nrmse =' + str(psp_fits.nrmse()))
#             mplt.show()

# def save_fit_psp_test_set():
#     """NOTE THIS CODE DOES NOT WORK BUT IS HERE FOR DOCUMENTATION PURPOSES SO 
#     THAT WE CAN TRACE BACK HOW THE TEST DATA WAS CREATED IF NEEDED.
#     Create a test set of data for testing the fit_psp function.  Uses Steph's 
#     original first_puls_feature.py code to filter out error causing data.
    
#     Example run statement
#     python save save_fit_psp_test_set.py --organism mouse --connection ee
    
#     Comment in the code that does the saving at the bottom
#     """
    
    
#     import pyqtgraph as pg
#     import numpy as np
#     import csv
#     import sys
#     import argparse
#     from multipatch_analysis.experiment_list import cached_experiments
#     from manuscript_figures import get_response, get_amplitude, response_filter, feature_anova, write_cache, trace_plot, \
#         colors_human, colors_mouse, fail_rate, pulse_qc, feature_kw
#     from synapse_comparison import load_cache, summary_plot_pulse
#     from neuroanalysis.data import TSeriesList, TSeries
#     from neuroanalysis.ui.plot_grid import PlotGrid
#     from multipatch_analysis.connection_detection import fit_psp
#     from rep_connections import ee_connections, human_connections, no_include, all_connections, ie_connections, ii_connections, ei_connections
#     from multipatch_analysis.synaptic_dynamics import DynamicsAnalyzer
#     from scipy import stats
#     import time
#     import pandas as pd
#     import json
#     import os
    
#     app = pg.mkQApp()
#     pg.dbg()
#     pg.setConfigOption('background', 'w')
#     pg.setConfigOption('foreground', 'k')
    
#     parser = argparse.ArgumentParser(description='Enter organism and type of connection you"d like to analyze ex: mouse ee (all mouse excitatory-'
#                     'excitatory). Alternatively enter a cre-type connection ex: sim1-sim1')
#     parser.add_argument('--organism', dest='organism', help='Select mouse or human')
#     parser.add_argument('--connection', dest='connection', help='Specify connections to analyze')
#     args = vars(parser.parse_args(sys.argv[1:]))
    
#     all_expts = cached_experiments()
#     manifest = {'Type': [], 'Connection': [], 'amp': [], 'latency': [],'rise':[], 'rise2080': [], 'rise1090': [], 'rise1080': [],
#                 'decay': [], 'nrmse': [], 'CV': []}
#     fit_qc = {'nrmse': 8, 'decay': 499e-3}
    
#     if args['organism'] == 'mouse':
#         color_palette = colors_mouse
#         calcium = 'high'
#         age = '40-60'
#         sweep_threshold = 3
#         threshold = 0.03e-3
#         connection = args['connection']
#         if connection == 'ee':
#             connection_types = ee_connections.keys()
#         elif connection == 'ii':
#             connection_types = ii_connections.keys()
#         elif connection == 'ei':
#             connection_types = ei_connections.keys()
#         elif connection == 'ie':
#             connection_types == ie_connections.keys()
#         elif connection == 'all':
#             connection_types = all_connections.keys()
#         elif len(connection.split('-')) == 2:
#             c_type = connection.split('-')
#             if c_type[0] == '2/3':
#                 pre_type = ('2/3', 'unknown')
#             else:
#                 pre_type = (None, c_type[0])
#             if c_type[1] == '2/3':
#                 post_type = ('2/3', 'unknown')
#             else:
#                 post_type = (None, c_type[0])
#             connection_types = [(pre_type, post_type)]
#     elif args['organism'] == 'human':
#         color_palette = colors_human
#         calcium = None
#         age = None
#         sweep_threshold = 5
#         threshold = None
#         connection = args['connection']
#         if connection == 'ee':
#             connection_types = human_connections.keys()
#         else:
#             c_type = connection.split('-')
#             connection_types = [((c_type[0], 'unknown'), (c_type[1], 'unknown'))]
    
#     plt = pg.plot()
    
#     scale_offset = (-20, -20)
#     scale_anchor = (0.4, 1)
#     holding = [-65, -75]
#     qc_plot = pg.plot()
#     grand_response = {}
#     expt_ids = {}
#     feature_plot = None
#     feature2_plot = PlotGrid()
#     feature2_plot.set_shape(5,1)
#     feature2_plot.show()
#     feature3_plot = PlotGrid()
#     feature3_plot.set_shape(1, 3)
#     feature3_plot.show()
#     amp_plot = pg.plot()
#     synapse_plot = PlotGrid()
#     synapse_plot.set_shape(len(connection_types), 1)
#     synapse_plot.show()
#     for c in range(len(connection_types)):
#         cre_type = (connection_types[c][0][1], connection_types[c][1][1])
#         target_layer = (connection_types[c][0][0], connection_types[c][1][0])
#         conn_type = connection_types[c]
#         expt_list = all_expts.select(cre_type=cre_type, target_layer=target_layer, calcium=calcium, age=age)
#         color = color_palette[c]
#         grand_response[conn_type[0]] = {'trace': [], 'amp': [], 'latency': [], 'rise': [], 'dist': [], 'decay':[], 'CV': [], 'amp_measured': []}
#         expt_ids[conn_type[0]] = []
#         synapse_plot[c, 0].addLegend()
#         for expt in expt_list:
#             for pre, post in expt.connections:
#                 if [expt.uid, pre, post] in no_include:
#                     continue
#                 cre_check = expt.cells[pre].cre_type == cre_type[0] and expt.cells[post].cre_type == cre_type[1]
#                 layer_check = expt.cells[pre].target_layer == target_layer[0] and expt.cells[post].target_layer == target_layer[1]
#                 if cre_check is True and layer_check is True:
#                     pulse_response, artifact = get_response(expt, pre, post, analysis_type='pulse')
#                     if threshold is not None and artifact > threshold:
#                         continue
#                     response_subset, hold = response_filter(pulse_response, freq_range=[0, 50], holding_range=holding, pulse=True)
#                     if len(response_subset) >= sweep_threshold:
#                         qc_plot.clear()
#                         qc_list = pulse_qc(response_subset, baseline=1.5, pulse=None, plot=qc_plot)
#                         if len(qc_list) >= sweep_threshold:
#                             avg_trace, avg_amp, amp_sign, peak_t = get_amplitude(qc_list)
#     #                        if amp_sign is '-':
#     #                            continue
#     #                        #print(('%s, %0.0f' %((expt.uid, pre, post), hold, )))
#     #                        all_amps = fail_rate(response_subset, '+', peak_t)
#     #                        cv = np.std(all_amps)/np.mean(all_amps)
#     #                        
#     #                        # weight parts of the trace during fitting
#                             dt = avg_trace.dt
#                             weight = np.ones(len(avg_trace.data))*10.  #set everything to ten initially
#                             weight[int(10e-3/dt):int(12e-3/dt)] = 0.   #area around stim artifact
#                             weight[int(12e-3/dt):int(19e-3/dt)] = 30.  #area around steep PSP rise 
                            
#                             # check if the test data dir is there and if not create it
#                             test_data_dir='test_psp_fit'
#                             if not os.path.isdir(test_data_dir):
#                                 os.mkdir(test_data_dir)
                                
#                             save_dict={}
#                             save_dict['input']={'data': avg_trace.data.tolist(),
#                                                 'dtype': str(avg_trace.data.dtype),
#                                                 'dt': float(avg_trace.dt),
#                                                 'amp_sign': amp_sign,
#                                                 'yoffset': 0, 
#                                                 'xoffset': 14e-3, 
#                                                 'avg_amp': float(avg_amp),
#                                                 'method': 'leastsq', 
#                                                 'stacked': False, 
#                                                 'rise_time_mult_factor': 10., 
#                                                 'weight': weight.tolist()} 
                            
#                             # need to remake trace because different output is created
#                             avg_trace_simple=TSeries(data=np.array(save_dict['input']['data']), dt=save_dict['input']['dt']) # create TSeries object
                            
#                             psp_fits_original = fit_psp(avg_trace, 
#                                                sign=save_dict['input']['amp_sign'], 
#                                                yoffset=save_dict['input']['yoffset'], 
#                                                xoffset=save_dict['input']['xoffset'], 
#                                                amp=save_dict['input']['avg_amp'],
#                                                method=save_dict['input']['method'], 
#                                                stacked=save_dict['input']['stacked'], 
#                                                rise_time_mult_factor=save_dict['input']['rise_time_mult_factor'], 
#                                                fit_kws={'weights': save_dict['input']['weight']})  
    
#                             psp_fits_simple = fit_psp(avg_trace_simple, 
#                                                sign=save_dict['input']['amp_sign'], 
#                                                yoffset=save_dict['input']['yoffset'], 
#                                                xoffset=save_dict['input']['xoffset'], 
#                                                amp=save_dict['input']['avg_amp'],
#                                                method=save_dict['input']['method'], 
#                                                stacked=save_dict['input']['stacked'], 
#                                                rise_time_mult_factor=save_dict['input']['rise_time_mult_factor'], 
#                                                fit_kws={'weights': save_dict['input']['weight']})  
#                             print(expt.uid, pre, post    )
#                             if psp_fits_original.nrmse()!=psp_fits_simple.nrmse():     
#                                 print('  the nrmse values dont match')
#                                 print('\toriginal', psp_fits_original.nrmse())
#                                 print('\tsimple', psp_fits_simple.nrmse())
    
    
                            
#     #                        save_dict['out']={}
#     #                        save_dict['out']['best_values']=psp_fits.best_values     
#     #                        save_dict['out']['best_fit']=psp_fits.best_fit.tolist()
#     #                        save_dict['out']['nrmse']=float(psp_fits.nrmse())
#     #                        with open(os.path.join(test_data_dir,expt.uid + '_' + str(pre) + '_' + str(post)+'NOTstacked.json'), 'w') as out_file:
#     #                            json.dump(save_dict, out_file)


            
# if __name__== "__main__":

# #    check_psp_fitting() #use this to diagnose how fits differ from test data
#     test_psp_fitting()
