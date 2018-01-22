# -*- coding: utf-8 -*-
# Copyright (C) Derek Davis (2017)
#
# This file is part of Detchar Cleaning Code.

import numpy as np
from numpy.fft import rfft, irfft
import scipy.signal as sig
import h5py
from pycbc import types,frame,strain
import pycbc.filter.resample as sample


def inner_product(data1,data2):
    """This takes two timeseries and calculates an inner product
    The output is a frequency series which is approx
    a transfer function (up to a constant)
    """
    assert data1.duration==data2.duration
    assert data1.start_time==data2.start_time
    srate1=data1.sample_rate
    srate2=data2.sample_rate
    fdata1=rfft(sig.hann(len(data1))*data1.detrend(type='constant').numpy())
    fdata2=rfft(sig.hann(len(data2))*data2.detrend(type='constant').numpy())
    max_idx=min(len(fdata1),len(fdata2))
    return types.frequencyseries.FrequencySeries(fdata1[:max_idx]*fdata2.conjugate()[:max_idx],
                            delta_f=1./data1.duration,epoch=data1.start_time)

def avg_freq_bins(fdata,new_df,tol=2.5e-9):
    """This funtion averages near frequency bins
    Takes in and outputs a FreqSeries
    """
    nyquist=(len(fdata)-1)*fdata.delta_f
    new_flen=1+int(nyquist/float(new_df))
    binlen=int(new_df/float(fdata.delta_f))
    win=sig.hann(2*binlen)
    win/=np.sum(win)
    result=np.zeros(new_flen,dtype=fdata.dtype)
    #This checks to ensure the binning will be exact
    assert (fdata.duration % binlen) == 0
    for idx in range(1,new_flen-1):
        result[idx]=np.sum(win*fdata.numpy()[binlen*(idx-1):binlen*(idx+1)])
        #result[idx]=np.sum(fdata.numpy()[binlen*(idx-1):binlen*(idx+1)])
    result[np.absolute(result) < max(np.absolute(result))*tol] = 0
    return types.frequencyseries.FrequencySeries(result,delta_f=new_df,
                            epoch=fdata.start_time)

def avg_inner_product(d1, d2, new_df=1.,tol=2.5e-9):
    """This wraps the two previous functions
    """
    assert d1.duration==d2.duration
    assert d1.start_time==d2.start_time
    if new_df > d1.delta_f:
        out = avg_freq_bins(inner_product(d1, d2), new_df,tol=tol)
    else:
        out_np = (inner_product(d1, d2)).numpy()
        out_np[np.absolute(out_np) < max(np.absolute(out_np))*tol] = 0
        out = types.frequencyseries.FrequencySeries(out_np,delta_f=d1.delta_f,
                            epoch=d1.start_time)
    return out


def measure_tf(target, witness, df=1.):
    """Function to evaluate the tf of a single
    witness channel. 
    Is not used in primary scripts.
    """
    iprod1=avg_inner_product(target,witness, df,tol=0)
    iprod2=avg_inner_product(witness,witness, df,tol=0)
    max_idx=min(len(iprod1),len(iprod2))
    result = np.divide(iprod1[:max_idx],iprod2[:max_idx])
    result[0] = 0.
    result[-1] = 0.
    return types.frequencyseries.FrequencySeries(result,
                            delta_f=df,epoch=target.start_time)

def project(target, witness, tf):
    """Similar to subtract function.
    Removes need to include 0*data in call.
    Not used in primary scripts.
    """
    assert target.duration == witness.duration
    assert target.start_time == witness.start_time

    srate=int(target.sample_rate)
    tlen=len(target)
    flen=1+tlen/2

    tmp=irfft(tf)
    tmp_len=len(tmp)
    tmp=sig.hann(tmp_len)*np.roll(tmp, tmp_len/2)
    tmp.resize(len(witness)) # pad with zeros to length of witness
    tf_long=rfft(np.roll(tmp, -tmp_len/2))

    tmp=rfft(witness.numpy().astype(np.float64))*tf_long
    tmp.resize(flen)

    result = irfft(tmp)

    pad=int(np.ceil(0.5/tf.delta_f))
    return types.timeseries.TimeSeries(result[int(pad*srate):-int(pad*srate)],
                        delta_t=1./target.sample_rate,
                        epoch=target.start_time+pad)


def subtract(target, witness, tf):
    """Subtracts noise from target channel
    using witness timeseries and relevant
    transfer function.
    """
    assert target.duration == witness.duration
    assert target.start_time == witness.start_time

    srate=int(target.sample_rate)
    pad=int(np.ceil(0.5/tf.delta_f))
    target_pad = target.copy()
    witness_pad = witness.copy()

    tlen=len(target_pad)
    flen=1+tlen/2

    tmp=irfft(tf)
    tmp_len=len(tmp)
    tmp=sig.hann(tmp_len)*np.roll(tmp, tmp_len/2)
    tmp.resize(len(witness_pad)) # pad with zeros to length of witness
    tf_long=rfft(np.roll(tmp, -tmp_len/2))

    tmp=rfft(witness_pad.numpy().astype(np.float64))*tf_long
    tmp.resize(flen)

    #result=target.numpy()-irfft((srate/float(witness.sample_rate))*tmp)
    result=target_pad.numpy()-irfft(tmp)

    return types.timeseries.TimeSeries(result[int(pad*srate):-int(pad*srate)],
                        delta_t=1./target.sample_rate,
                        epoch=target.start_time+pad)

def clean_data(data,aux_data,trans_func):
    """subtracts noise from a given strain channel using a 
    list of auxilary channels. 

    Aux channels and related transfer functions given in the form
    of lists.
    
    Wrapper around subtract() function
    """
    assert len(aux_data) == len(trans_func)
    data_clean=data_start=subtract(data, 0.*aux_data[0], trans_func[0])
    for i in range(len(aux_data)):
        inv_noise = subtract(0.*data, aux_data[i], trans_func[i])
        data_clean=data_clean+inv_noise
    data_subtracted = data_start - data_clean
    return data_clean, data_subtracted

def bp_chunk(ts, new_srate, f_low, f_high):
    """Bandpass function on timeseries. 
    Also resamples if needed.
    """
    srate=int(ts.sample_rate)
    if (f_high > srate/2):
        print "high frequency too high, choosing something lower..."
        f_high = int(srate/2*.9)
    if (srate < new_srate):
        print "sample rate too low, padding..."
        tmp_f=rfft(ts.numpy())
        tmp_f.resize(int(new_srate*ts.duration/2)+1) # pad with zeros to length of witness
        tmp_ts = irfft(tmp_f)
    else:
        tmp_ts = ts.numpy()
    bp=sig.firwin(4*srate,[f_low, f_high],nyq=srate/2.,window='hann',pass_zero=False)
    bp.resize(len(tmp_ts))
    #tmp=(new_srate/float(srate))*abs(rfft(bp))*rfft(tmp_ts)
    tmp=abs(rfft(bp))*rfft(tmp_ts)
    padidx=2*int(new_srate) # Remove 2 seconds of corrupted data from each end
    return types.timeseries.TimeSeries(
                        irfft(tmp[:1+int(ts.duration*new_srate/2)])[padidx:-padidx],
                        delta_t=1./new_srate,epoch=ts.start_time+2)

def _split_thirds(somearr):
    skip=len(somearr)/3
    return np.concatenate([somearr[:skip],somearr[-skip:]])

def _smooth_mid(myarr):
    """Smoothing function. 
    Used to interpolate between missing data points.
    Main application is to to estimate the transfer
    function when a line obscures the data.
    """
    skip=len(myarr)/3
    xarr=np.linspace(-1.,1.,3*skip)
    a,b,c=np.polyfit(_split_thirds(xarr),
                     _split_thirds(myarr),
                     2)
    old_arr = myarr.copy()
    myarr[skip:-skip]=types.Array(a*xarr[skip:-skip]*xarr[skip:-skip]+b*xarr[skip:-skip]+c)
    return myarr
    
def clean_tf(mytf,f1,f2):
    """Removes impact of lines on transfer function. 
    Does this via removal of corrupted bins and
    then interpolating over that region.
    """
    assert f2 > f1
    assert f2 < mytf.sample_rate/2
    df=mytf.delta_f
    idx1,idx2 = int(f1/df), int(f2/df)
    width=idx2-idx1
    idx0,idx3 = idx1-width, idx2+width
    
    result=mytf.copy()
    
    result[idx0:idx3] = _smooth_mid(result.real()[idx0:idx3])+ 1j*_smooth_mid(result.imag()[idx0:idx3])
    
    return result

def remove_line(tf,line_tuples):
    """Removes impact of multiple lines. 
    Affected regions given as a list of tuples. 

    Wrapper around clean_tf() function
    """
    tmp = tf
    for low,high in line_tuples:
        tmp=clean_tf(tmp,low,high)
    return tmp

def correlation_matrix(aux_data,chan_list,df=1.,tol=2.5e9):
    """Sets up the correlation matrix for a group of channels.
    The correlation matric is given by 
        M_ij = <n_i,n_j>, 
    the inner product of channels i and j. 

    A matrix is calculated for each frequency bin and 
    is stored in a dictionary.
    """
    freq_max = int((aux_data[0].sample_rate)/(2))
    freq_len = int(freq_max / df) + 1
    dim = len(aux_data)
    corr_list = []
    for i in range(freq_len):
        corr_list.append(np.identity(dim,dtype=complex))
    for i in range(dim):
        for j in range(dim):
            cross_tf = avg_inner_product(aux_data[i], aux_data[j],new_df=df, tol=tol)
            for k in range(freq_len):
                corr_list[k][j,i]= cross_tf[k]
    corr_dict = dict(zip(np.linspace(0,freq_max,num=freq_len),corr_list))
    chan_dict = dict(zip(range(0,dim,1),chan_list))
    return corr_dict, chan_dict

def detangle_tf(corr_dict, chan_dict, tf_list,max_freq=2000):
    """ Calculates the pseudoinverse of a given correlation matrix.
    The use of a pseudoinverse resolves need to edit the given matrix
    when a eigenvalue of 0 is given. The most common example of such is
    when a channel has no frequency content. 

    To prevent the removal of marginally corelated noise from a channel 
    that has no content, the entire row and column of such a channel is
    zeroed. 
    """
    dim = len(tf_list)
    freq_len = len(tf_list[0])
    max_element = int(max_freq/tf_list[0].delta_f)
    corr_matrix = []
    for freq in np.sort(corr_dict.keys()):
        corr_matrix.append(corr_dict[freq])
    inv_corr_list = []
    tf_val = []
    detangle_list = []
    for k in range(freq_len):
        tf_val.append(np.zeros(dim,dtype=complex))
        for i in range(dim):
            tf_val[k][i] = tf_list[i][k]
    for k in range(freq_len):
        if (k < 10) or (k == freq_len-1) or (k > max_element):
            detangle_list.append(np.zeros(dim))
            #inv_matrix = np.zeros((dim,dim))
        else:
            for i in range(dim):
                if (corr_matrix[k][i,i] == 0) or tf_val[k][i] == 0:
                    corr_matrix[k][i] = corr_matrix[k][:,i] = 0
            inv_matrix = np.linalg.pinv(corr_matrix[k])
            detangle_list.append(inv_matrix.dot(tf_val[k]))
    final_tf_list = []
    for i in range(dim):
        final_tf = np.zeros(freq_len,dtype=complex)
        for k in range(freq_len):
            final_tf[k] = detangle_list[k][i]
        final_tf_list.append(types.frequencyseries.FrequencySeries(
                             final_tf,delta_f=tf_list[i].delta_f,
                             epoch=tf_list[i].start_time))
    return final_tf_list

def apply_gate(timeseries, gate_params):
    """Applies pycbc gate to timeseries
    This is done as one line to keep dependencies in this file
    """
    gated_timeseries = strain.gate_data(timeseries, gate_params)
    return gated_timeseries

def read_in(start, end, aux_chan_list, cut=1, ifo='H1',
            strain_frame_type='H1_HOFT_C00',
            strain_channel='GDS-CALIB_STRAIN',
            no_strain=False):
    """Reads in the needed set of strain and aux channels. 

    Output as single timeseries and list of timeseries. 
    """
    start_get = start - cut
    end_get = end + cut
    if no_strain == False:
        data = frame.query_and_read_frame(strain_frame_type, 
               '%s'%(strain_channel), start_get, end_get,sieve='hdfs')
    else:
        data = None
    aux = []
    channels = ['%s' % (chan) for chan in aux_chan_list]
    multitimeseries = frame.query_and_read_frame(ifo + "_R",
                                    channels,
                                    start_get, end_get,sieve='hdfs')
    return data, multitimeseries

def add_buffer(clean_data_chunk, prev_buffer=None, dur=1024,
               write=False, write_size=4, data_dir='./', ifo='H1',
               strain_type='C00',output_channel='CLEAN_STRAIN', 
               frame_tag='HOFT_CLEANED'):
    """ Takes in strain timeseries and overlap adds to previous buffer. 

    Once a maximal timeseries duration is reached, the data is flushed
    and written to a frame file. 

    Overlap-add is accomplished via a hann windowing on each section. 
    """
    strain_sample_rate = clean_data_chunk.sample_rate
    phase = np.linspace(0, np.pi/2, num=strain_sample_rate*dur/2)

    start = clean_data_chunk.start_time
    end = start + dur / 2
    if prev_buffer is None: 
        data_buffer = clean_data_chunk.time_slice(start, end).copy()
    else:
        prev_end = prev_buffer.end_time - dur / 2
        hann_start = types.timeseries.TimeSeries(np.sin(phase)**2, 
                         delta_t=1.0/strain_sample_rate,epoch=prev_end)
        hann_end = types.timeseries.TimeSeries(np.cos(phase)**2, 
                         delta_t=1.0/strain_sample_rate,epoch=start)
        data1 = prev_buffer.time_slice(prev_end, prev_buffer.end_time) * hann_end
        data2 = clean_data_chunk.time_slice(start, end) * hann_start

        if prev_buffer.duration > dur/2 :
            data_buffer = prev_buffer.time_slice(prev_buffer.start_time, prev_end).copy()
            olen = len(data_buffer)
            data_buffer.resize(olen + len(data1))
            data_buffer[olen:] = data1 + data2
        else:
            data_buffer = data1 + data2
    if clean_data_chunk.end_time > end:
        chunk_tail = clean_data_chunk.time_slice(end, clean_data_chunk.end_time)
    else:
        chunk_tail = []
    max_dur = dur * write_size
    print "buffer duration currently", data_buffer.duration, "out of a max", max_dur

    if data_buffer.duration == max_dur:
        if write==True:
            fname = "%s/%s-%s_%s-%u-%u.gwf" % (data_dir,
                       ifo, strain_type, frame_tag, 
                       int(data_buffer.start_time), 
                       int(data_buffer.duration))
            channel = '%s'%(output_channel)
            frame.write_frame(fname, channel, data_buffer)
        data_buffer = chunk_tail
    else:
        olen = len(data_buffer)
        data_buffer.resize(olen + len(chunk_tail))
        data_buffer[olen:] = chunk_tail
    return data_buffer

def create_tf_hdf(file_path,start_list,aux_chan_list,duration_tf,
                  filt_len, ifo='H1'):
    """Sets up hdf file to store transfer functions

    Used in conjunction with append_tf_hdf()
    """
    hdf_file = h5py.File(file_path,'x')
    ifo_group = hdf_file.create_group(ifo)
    ifo_group.create_dataset("times",data=start_list)
    ifo_group.create_dataset("duration",data=[duration_tf])
    ifo_group.create_dataset("channels",data=aux_chan_list)
    ifo_group.create_dataset("filter_length",data=[filt_len])
    for time in start_list:
        ifo_group.create_group('tfs/'+str(time))
    hdf_file.close()

def append_tf_hdf(file_path,tf_start,aux_chan_list,det_tf,ifo="H1"):
    """Adds list of tiransfer functions and related metadata to 
    previously created hdf file. 

    Used in conjunction with create_tf_hdf()
    """
    hdf_file = h5py.File(file_path,'r+')
    tf_dict = dict(zip(aux_chan_list,det_tf))
    group_exists = ('%s/tfs/%s'%(ifo,str(tf_start))) in hdf_file
    if group_exists == False:
        hdf_file.create_group('%s/tfs/%s'%(ifo,str(tf_start)))
    time_group = hdf_file['%s/tfs/%s'%(ifo,str(tf_start))]
    for i in range(0,len(aux_chan_list)):
        tf_dset = time_group.create_dataset('%s'%(aux_chan_list[i]),data=det_tf[i])
        tf_dset.attrs['epoch'] = float(det_tf[i].start_time)
        tf_dset.attrs['delta_f'] = det_tf[i].delta_f
    hdf_file.close()

def read_tf_hdf(tf_file_path,start_data,end_data,ifo='H1'):
    """Reads in hdf file with sets of transfer functions to 
    use in cleaning process. 

    Also sets up parameters of clenaing including 
    start times
    duration
    filter length
    channel listing
    """
    tf_file = h5py.File(tf_file_path)
    tf_times = tf_file['%s/times'%ifo][:]
    aux_chan_list = tf_file['%s/channels'%ifo][:]
    duration_tf = tf_file['%s/duration'%ifo][0]
    filter_length = tf_file['%s/filter_length'%ifo][0]
    start_list = []
    tf_dict = {}
    start_pad = start_data #+ np.ceil(.5*filter_length)
    end_pad = end_data #- np.ceil(.5*filter_length)
    print "The tf times are:", tf_times
    for time in tf_times:
        if (start_pad <= time < end_pad) or (start_pad < time+duration_tf <= end_pad):
            tf_list = []
            for chan in aux_chan_list:
                tf_dset = tf_file['%s/tfs/%s/%s'%(ifo,str(time),chan)]
                chan_tf = types.frequencyseries.FrequencySeries(tf_dset[:],
                                epoch=tf_dset.attrs['epoch'],delta_f=tf_dset.attrs['delta_f'])
                tf_list.append(chan_tf)
            tf_dict[time] = tf_list
            start_list.append(time)
    #if start_list[-1]+duration_tf < end_pad:
    #    time = start_list[-1]
    #    tf_list = []
    #    for chan in aux_chan_list:
    #        tf_dset = tf_file['%s/tfs/%s/%s'%(ifo,str(time),chan)]
    #        chan_tf = types.frequencyseries.FrequencySeries(tf_dset[:],
    #                        epoch=tf_dset.attrs['epoch'],delta_f=tf_dset.attrs['delta_f'])
    #        tf_list.append(chan_tf)
    #    tf_dict[time+duration_tf/2] = tf_list
    #    start_list.append(time+duration_tf/2)
    if len(start_list) == 0:
        if tf_times[-1] < start_pad:
            time = tf_times[-1]
        else:
            time = tf_times[0]
        tf_list = []
        for chan in aux_chan_list:
            tf_dset = tf_file['%s/tfs/%s/%s'%(ifo,str(time),chan)]
            chan_tf = types.frequencyseries.FrequencySeries(tf_dset[:],
                            epoch=tf_dset.attrs['epoch'],delta_f=tf_dset.attrs['delta_f'])
            tf_list.append(chan_tf)
        tf_dict[start_pad] = tf_list
        start_list.append(start_pad)        
        duration_tf = min(duration_tf, end_pad-start_pad)
    return tf_dict,start_list,aux_chan_list,duration_tf,filter_length



    

