import numpy as np

#%% findPeaksTroughs: Find local maxima and minima. This is a custom version of MATLAB's findpeaks().

def findPeaksTroughs(vctr):

#   Inputs:
#
#      - vctr: Input signal vector.
#
#   Outputs:
#
#      - pks:	Values of peaks or troughs identified. Currently does
#               *not* check for Inf values or plateaus. Signal end-points
#               are excluded.
#      - locs:	Indices where the peaks occur.
#      - wds:	'Width' of each peak, defined as distance from previous
#               peak to next peak. This is DIFFERENT from MATLAB function.
#      - prms:	'Prominence' of each peak, defined by the min() of the
#               height difference between current peak, and previous 
#               and next peaks.

#   Author:         Arnab Gupta
#                   Ph.D. Candidate, Virginia Tech.
#                   Blacksburg, VA.
#   Website:        http://arnabocean.com
#   Repository:     http://bitbucket.org/arnabocean
#   Email:          arnab@arnabocean.com
#
#   Version:        2.0
#   Last Revised:   Thursday, May 18, 2017
    
#   Translated from MATLAB to Python 3 by Francisco Rotea (September, 2019)
#   Repository:     https://github.com/franciscorotea/wave-mode-arrivals/
#   Email:          francisco.rotea@gmail.com

##	Find diff of vector elements; create second diff vector offset by 1
    
    dff = np.diff(vctr)

    dff1 = np.zeros((len(dff)+1))
    dff1[1:len(dff)+1] = dff

    dff = np.append(dff, 0)

##	Find product of successive elements of diff vector; 
##	implemented here by element-wise multiplication of dff and dff1

    mlt = dff*dff1
    mlt = np.delete(mlt, 0)                 # Extra
    mlt = np.delete(mlt, -1)                # Extra

##	Find indices of interest

    idx = np.array(np.where(mlt <= 0))      # Multiplication results in <=0 only for peaks or troughs
    locs = idx+1                            # Needs offset because 'diff' operation decreases length by 1
    pks = vctr[locs]    

##	Prepare to calculate prominence and widths

    locs1 = np.ones((len(idx[0,:])+2))      # Pad ends with start and end values of vctr
    locs1[1:-1] = locs
    locs1[-1] = len(vctr)   

    pks1 = np.ones((len(idx[0,:])+2))       # Pad ends with start and end values of vctr
    pks1[1:-1] = pks
    pks1[0] = vctr[0]
    pks1[-1] = vctr[-1]

    wds = np.zeros((len(pks1)-1))
    prms = np.zeros((len(pks1)-1))

##	Calculate prominence and width values by cycling through vctor

    for jj in np.arange(1,len(locs1)-1):
        
        ##	Calculate width values
        
        wds[jj] = locs1[jj+1] - locs1[jj-1]
        
        ##	Calculate left and right heights; find min() value
        
        lfp = abs(pks1[jj] - pks1[jj-1])
        rtp = abs(pks1[jj+1] - pks1[jj])

        prms[jj] = min(lfp, rtp)

##	Remove excess elements introduced earlier

    wds = np.delete(wds, 0)
    prms = np.delete(prms, 0)
    locs1 = np.delete(locs1, 0)
    pks1 = np.delete(pks1, 0)
    locs1 = np.delete(locs1, -1)
    pks1 = np.delete(pks1, -1)

    return pks1, locs1, wds, prms

#%% findSignalThreshold: Identify threshold value where signal amplitude changes.

def findSignalThreshold(vctr):
    
#   Inputs:
#
#      - vctr: Input vector
#
#   Outputs:
#
#      - thresh: Threshold value of amplitude calculated

#   Author:         Arnab Gupta
#                   Ph.D. Candidate, Virginia Tech.
#                   Blacksburg, VA.
#   Website:        http://arnabocean.com
#   Repository:     http://bitbucket.org/arnabocean
#   Email:          arnab@arnabocean.com

#   Version:        1.0
#   Last Revised:   Thursday, May 18, 2017

#   Translated from MATLAB to Python 3 by Francisco Rotea (September, 2019)
#   Repository:     https://github.com/franciscorotea/wave-mode-arrivals/
#   Email:          francisco.rotea@gmail.com
    
##	Parameters
    
    strt = 0
    endd = 25
    incr = 20

    noiseMin = 1e-6
    upcntLim = 2

    thresh = []

    flag1 = 1
    flag2 = 0

    upcnt = 0

    while flag1 == 1:
        
	##	Find mean value of portion of signal
    
        if thresh:
            tmpthresh = thresh                      # Save previous value temporarily

        thresh = np.mean(vctr[strt:endd])

        strt = strt + incr
        endd = endd + incr 
        
	##	Decide if current thresh value meets certain criteria
    
        if thresh > noiseMin:                       # We want this condition to be true successively, >upcntlim times
            
            flag2 = 1
            upcnt = upcnt + 1
                    
            if upcnt > 1:                           # Restore previously saved value, i.e. the first time when thresh > noisemin occurred

                thresh = tmpthresh

        elif thresh <= noiseMin and flag2 == 1:     # If "successive" streak expires, reset counters

            flag2 = 0
            upcnt = 0

	##	End while loop

        if endd > len(vctr) or (thresh > noiseMin and upcnt > upcntLim):
            
            flag1 = 0

    return thresh

#%% getExtensionArrival: Detect the index of signal when extensional wave mode arrives.

def getExtensionArrival(sgnl, noisemult = 10, noisemin = 1e-6):
    
#   Inputs:
#
#      - sgnl: Input vector of ultrasonic acoustic wave.
#      - noisemult: (Optional) Threshold multiplier, used to multiply 
#                   the initial noise portion to signal to determine 
#                   working threshold. Default value: 10.
#      - noisemin: (Optional) Absolute minimum noise value, to be used
#                  if actual noise content of signal is negligible.
#                  Default value: 1E-6
#
#   Outputs:
#
#      - extIDX: Index value of sgnl where extensional wave mode is estimated
#                to have arrived.

#   Author:         Arnab Gupta
#                   Ph.D. Candidate, Virginia Tech.
#                   Blacksburg, VA.
#   Website:        http://arnabocean.com
#   Repository:     http://bitbucket.org/arnabocean
#   Email:          arnab@arnabocean.com
#
#   Version:        1.0
#   Last Revised:   Wednesday, May 31, 2017

#   Translated from MATLAB to Python 3 by Francisco Rotea (September, 2019)
#   Repository:     https://github.com/franciscorotea/wave-mode-arrivals/
#   Email:          francisco.rotea@gmail.com
    
##  Find Peaks and Troughs in signal
    
    pks, locs, wds, prominence = findPeaksTroughs(sgnl)

##  Calculate paramater that scales peak values with a prominence value weight. Also normalize the parameter.

    param = np.abs(pks*prominence)
    nrm = np.sort(param)[::-1]

    if nrm[0]/nrm[1] > 5:
        param = param/nrm[1]                # If there is one single very large peak, then use 2nd highest peak to normalize.
    else:
        param = param/nrm[0]                # Else use the largest peak to normalize

##  Calculate reference values of noise and amplitude threshold to test signal amplitude against

    tmpidx = np.array(np.where(param == 1))
    tmpidx = int(tmpidx[0, 0])

    nnlim = max(np.floor(locs[tmpidx]/4), np.floor(len(sgnl)/40));

    nnlocs = np.array(np.where(locs <= nnlim))
    nnlocs = int(nnlocs[0, -1]) # revisar estos indices

    refnoisemean = max(np.mean(param[1:nnlocs]), noisemin);
    refmaxthresh = max(noisemult*refnoisemean, max(param[1:nnlocs]));

##  Find mean signal value until certain conditions are met (if conditions are met before signal ends, loop can be ended quicker)

    tstmean = []
    cnt = 0
    strt = 0
    endd = int(strt + np.floor(nnlocs/2))
    flag = 1

    while flag == 1:
        
        tstmean.append((strt, np.mean(param[strt:endd])))
        strt = endd + 1
        endd = int(strt + np.floor(nnlocs/2))

        if endd > len(param) or tstmean[cnt][1] > refmaxthresh:

            flag = 0

        cnt = cnt + 1
        
    tstmean = np.vstack(tstmean)

##  Make best guess for extIDX when the mean signal value NEVER meets the condition in loop above (i.e. loop ends when signal ends)

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
            
##  Zoom into portion of signal where extensional mode definitely arrives

    else:
        
        idx = np.array(np.where(tstmean[:, 1] < refnoisemean))
        idx = int(idx[0, -1])
        idx = int(tstmean[idx][0] - 1)

        param = np.delete(param, np.arange(0,idx+1))
        locs = np.delete(locs, np.arange(0,idx+1))

        idx = np.array(np.where(param > refmaxthresh))
        idx = int(idx[0, 0])

        param = np.delete(param, np.arange(idx+1, len(param)))
        locs = np.delete(locs, np.arange(idx+1, len(locs)))

##  Find best guess for extIDX from this small portion of the signal.

        idx = np.array(np.where(param <= refnoisemean))
        idx = int(idx[0, -1])

        extIDX = locs[idx + 1]

        if not extIDX:
            try:
                extIDX = locs[-2]
            except:
                extIDX = locs[-1]

    return int(extIDX)

#%% getFlexureArrival: Detect the index of signal when flexural wave mode arrives.

def getFlexureArrival(sgnl):

#   Inputs:
#
#      - sgnl: Input vector of ultrasonic acoustic wave.
#
#   Outputs:
#
#      - extIDX: Index value of sgnl where flexural wave mode is estimated
#                to have arrived.

#   Author:         Arnab Gupta
#                   Ph.D. Candidate, Virginia Tech.
#                   Blacksburg, VA.
#   Website:        http://arnabocean.com
#   Repository:     http://bitbucket.org/arnabocean
#   Email:          arnab@arnabocean.com
#
#   Version:        1.0
#   Last Revised:   Wednesday, May 31, 2017

#   Translated from MATLAB to Python 3 by Francisco Rotea (September, 2019)
#   Repository:     https://github.com/franciscorotea/wave-mode-arrivals/
#   Email:          francisco.rotea@gmail.com
    
##	Find Peaks and Troughs in signal
    
    pks, locs, wds, prominence = findPeaksTroughs(sgnl)

##	Calculate parameter that scales peak values with a prominence value weight. Also normalize the parameter.

    param = np.abs(pks*prominence)
    nrm = np.sort(param)[::-1]

    if nrm[0]/nrm[1] > 5:
        param = param/nrm[1]                # If there is one single very large peak, then use 2nd highest peak to normalize.
    else:
        param = param/nrm[0]                # Else use the largest peak to normalize
        
##  Evaluate appropriate threshold value

    thresh = findSignalThreshold(param)
    
##	Use threshold to identify significant peaks
    
    idx = np.array(np.where(param > thresh))
    idx = int(idx[0, 0])

##	Arrival time corresponds to first significant peak

    if not idx:
        flexIDX = 0
    else:
        flexIDX = locs[idx]

    return int(flexIDX)