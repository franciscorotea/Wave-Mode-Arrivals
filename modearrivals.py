"""Find time of arrival of flexural/extensional wave modes. For details 
about the algorithms implemented, please refer to:

Gupta, A., Duke, J. C., Identifying the arrival of extensional and 
flexural wave modes using Wavelet decomposition of ultrasonic signals, 
Ultrasonics 82, 261-271 (2018).

Author:         Arnab Gupta
                Ph.D. Candidate, Virginia Tech.
                Blacksburg, VA.
Website:        http://arnabocean.com
Repository:     http://bitbucket.org/arnabocean
Email:          arnab@arnabocean.com
Date:           May, 2017

Ported from MATLAB to Python 3 by Francisco Rotea.

Repository:     https://github.com/franciscorotea/
Email:          francisco.rotea@gmail.com
Date:           September, 2019

"""

import numpy as np

def find_peaks_troughs(vctr):

    """
    Find local maxima and minima. This is a custom version of MATLAB's 
    findpeaks() / Scipy's signal.find_peaks().

    Parameters
    ----------
    vctr : array_like
        Input signal vector.

    Returns
    -------
    pks : array_like
        Values of peaks or troughs identified. Currently does NOT check 
        for Inf values or plateaus. Signal end-points are excluded.
    locs : array_like
        Indices where the peaks occur.
    wds : array_like
        'Width' of each peak, defined as distance from previous peak to
        next peak.
    prms : array_like
        'Prominence' of each peak, defined by the min() of the height 
        difference between current peak, and previous and next peaks.

    """

    # Find diff of vector elements. Create second diff vector offset by 
    # 1.
    
    dff = np.diff(vctr)

    dff1 = np.zeros((len(dff)+1))
    dff1[1:len(dff)+1] = dff

    dff = np.append(dff, 0)

    # Find product of successive elements of diff vector.

    mlt = dff*dff1
    mlt = np.delete(mlt, 0)     # Extra
    mlt = np.delete(mlt, -1)    # Extra

    # Find indices of interest: multiplication result is <= 0 only for
    # peaks or troughs.

    idx = np.array(np.where(mlt <= 0))
    locs = idx+1  # `diff` decreases length by 1
    pks = vctr[locs]    

    # Prepare to calculate prominence and widths. Pad ends with start 
    # and end values of vctr.

    locs1 = np.ones((len(idx[0,:])+2))
    locs1[1:-1] = locs
    locs1[-1] = len(vctr)

    pks1 = np.ones((len(idx[0,:])+2))
    pks1[1:-1] = pks
    pks1[0] = vctr[0]
    pks1[-1] = vctr[-1]

    wds = np.zeros((len(pks1)-1))
    prms = np.zeros((len(pks1)-1))

    # Calculate prominence and width values by cycling through vector.

    for jj in np.arange(1, len(locs1)-1):
        
        # Calculate width values.
        
        wds[jj] = locs1[jj+1] - locs1[jj-1]
        
        # Calculate left and right heights and find min() value.
        
        lfp = np.abs(pks1[jj] - pks1[jj-1])
        rtp = np.abs(pks1[jj+1] - pks1[jj])

        prms[jj] = min(lfp, rtp)

    # Remove excess elements introduced earlier.

    wds = np.delete(wds, 0)
    prms = np.delete(prms, 0)
    locs1 = np.delete(locs1, 0)
    pks1 = np.delete(pks1, 0)
    locs1 = np.delete(locs1, -1)
    pks1 = np.delete(pks1, -1)

    return pks1, locs1, wds, prms


def find_signal_threshold(vctr):

    """
    Identify threshold value where signal amplitude changes.

    Parameters
    ----------
    vctr : array_like
        Input vector.

    Returns
    -------
    thresh : float
        Threshold value of amplitude calculated

    """

    # Initialization parameters.
    
    strt = 0
    endd = 25
    incr = 20

    noiseMin = 1e-6
    upcntLim = 2

    thresh = []

    flag1 = True
    flag2 = False

    upcnt = 0

    while flag1:
        
        # Find mean value of portion of signal.
    
        if thresh:
            tmpthresh = thresh      # Save previous value temporarily

        thresh = np.mean(vctr[strt:endd])

        strt = strt + incr
        endd = endd + incr 
        
        # Decide if current thresh value meets certain criteria.
    
        if thresh > noiseMin:       # We want this condition to be
                                    # true successively, >upcntlim times
            flag2 = True
            upcnt = upcnt + 1

            # Restore previously saved value,  i.e. the first time when 
            # thresh > noisemin occurred    

            if upcnt > 1:

                thresh = tmpthresh

        elif thresh <= noiseMin and flag2:    # If "successive" streak 
                                              # expires, reset counters
            flag2 = False
            upcnt = 0

        # End while loop.

        if endd > len(vctr) or (thresh > noiseMin and upcnt > upcntLim):
            
            flag1 = False

    return thresh


def get_extension_arrival(sgnl, noisemult = 10, noisemin = 1e-6):

    """
    Detect the index of signal when extensional wave mode arrives.

    Parameters
    ----------
    sgnl : array_like
        Input vector of ultrasonic acoustic wave.
    noisemult : float, optional
        Threshold multiplier, used to multiply the initial noise portion
        to signal to determine working threshold. Default value: 10.
    noisemin : float, optional
        Absolute minimum noise value, to be used if actual noise content
        of signal is negligible. Default value: 1E-6.

    Returns
    -------
    extIDX : int
        Index value of sgnl where extensional wave mode is estimated to
        have arrived.

    """
    
    # Find peaks and troughs in signal.
    
    pks, locs, wds, prominence = find_peaks_troughs(sgnl)

    # Calculate paramater that scales peak values with a prominence 
    # value weight. Also normalize the parameter.

    param = np.abs(pks*prominence)
    nrm = np.sort(param)[::-1]

    # If there is one single very large peak, then use 2nd highest peak 
    # to normalize. Else, use the largest peak.

    if nrm[0]/nrm[1] > 5:
        param = param/nrm[1]
    else:
        param = param/nrm[0]

    # Calculate reference values of noise and amplitude threshold to 
    # test signal amplitude against.

    tmpidx = np.array(np.where(param == 1))
    tmpidx = int(tmpidx[0, 0])

    nnlim = max(np.floor(locs[tmpidx]/4), np.floor(len(sgnl)/40))

    nnlocs = np.array(np.where(locs <= nnlim))
    nnlocs = int(nnlocs[0, -1])

    refnoisemean = max(np.mean(param[1:nnlocs]), noisemin)
    refmaxthresh = max(noisemult*refnoisemean, max(param[1:nnlocs]))

    # Find mean signal value until certain conditions are met (if 
    # conditions are met before signal ends, loop can be ended quicker).

    tstmean = []

    cnt = 0

    strt = 0
    endd = int(strt + np.floor(nnlocs/2))

    flag = True

    while flag:
        
        tstmean.append((strt, np.mean(param[strt:endd])))
        strt = endd + 1
        endd = int(strt + np.floor(nnlocs/2))

        if endd > len(param) or tstmean[cnt][1] > refmaxthresh:

            flag = False

        cnt = cnt + 1
        
    tstmean = np.vstack(tstmean)

    # Make best guess for extIDX when the mean signal value NEVER meets 
    # the condition in loop above (i.e. loop ends when signal ends).

    if tstmean[-1,1] <= refmaxthresh:
        
        idx = np.array(np.where(tstmean[:, 1] < refnoisemean))
        idx = int(idx[0, -1])
        
        if not idx:
            idx = 1
        
        param = np.delete(param, np.arange(0, int(tstmean[idx][0])+1))
        locs = np.delete(locs, np.arange(0, int(tstmean[idx][0])+1))

        idx = np.array(np.where(param > refnoisemean))
        idx = idx[0, 0]
                         
        if not idx:
            extIDX = 0
        else:
            extIDX = locs[idx]
            
    # Zoom into portion of signal where extensional mode definitely 
    # arrives.

    else:
        
        idx = np.array(np.where(tstmean[:, 1] < refnoisemean))
        idx = int(idx[0, -1])
        idx = int(tstmean[idx][0] - 1)

        param = np.delete(param, np.arange(0, idx+1))
        locs = np.delete(locs, np.arange(0, idx+1))

        idx = np.array(np.where(param > refmaxthresh))
        idx = int(idx[0, 0])

        param = np.delete(param, np.arange(idx+1, len(param)))
        locs = np.delete(locs, np.arange(idx+1, len(locs)))

        # Find best guess for extIDX from this small portion of the 
        # signal.

        idx = np.array(np.where(param <= refnoisemean))
        idx = int(idx[0, -1])

        extIDX = locs[idx + 1]

        if not extIDX:
            try:
                extIDX = locs[-2]
            except:
                extIDX = locs[-1]

    return int(extIDX)


def get_flexure_arrival(sgnl):

    """
    Detect the index of signal when flexural wave mode arrives.

    Parameters
    ----------
    sgnl : array_like
        Input vector of ultrasonic acoustic wave.

    Returns
    -------
    extIDX : int
        Index value of `sgnl` where flexural wave mode is estimated to
        have arrived.

    """
    
    # Find peaks and troughs in signal.
    
    pks, locs, wds, prominence = find_peaks_troughs(sgnl)

    # Calculate parameter that scales peak values with a prominence 
    # value weight. Also normalize the parameter.

    param = np.abs(pks*prominence)
    nrm = np.sort(param)[::-1]

    # # If there is one single very large peak, then use 2nd highest 
    # peak to normalize. Else, use the largest peak.

    if nrm[0]/nrm[1] > 5:
        param = param/nrm[1]
    else:
        param = param/nrm[0]
        
    # Evaluate appropriate threshold value.

    thresh = find_signal_threshold(param)
    
    # Use threshold to identify significant peaks.
    
    idx = np.array(np.where(param > thresh))
    idx = int(idx[0, 0])

    # Arrival time corresponds to first significant peak.

    if not idx:
        flexIDX = 0
    else:
        flexIDX = locs[idx]

    return int(flexIDX)