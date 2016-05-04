import numpy as np
import multiprocessing
import itertools


class FP_worker(multiprocessing.Process):
    """Worker class for farming out the work of doing the least
    squares fit

    """

    def __init__(self,
                 work_queue,
                 res_queue):
        """
        Work queue is a joinable queue, res_queue can be any sort of thing that supports put()
        """
        # background set up that must be done
        multiprocessing.Process.__init__(self)
        self.daemonic = True
        self.work_queue = work_queue
        self.res_queue = res_queue

    def run(self):
        """
        The assumption is that these will be run daemonic and reused for multiple work sessions
        """
        while True:
            work_lst = self.work_queue.get()
            if work_lst is None:          # poison pill
                return
            res_lst = []
            for x, y in work_lst:
                res_lst.append(fit_quad_to_peak(x, y))
            self.res_queue.put(res_lst)
            self.work_queue.task_done()


__PROCS = None
__WORK_QUEUE = None
__RES_QUEUE = None


def init_procs(N):
    """Sets up N processes"""
    global __PROCS
    global __WORK_QUEUE
    global __RES_QUEUE
    __WORK_QUEUE = multiprocessing.JoinableQueue()
    __RES_QUEUE = multiprocessing.Queue()
    __PROCS = [FP_worker(__WORK_QUEUE, __RES_QUEUE) for j in range(N)]
    for p in __PROCS:
        p.start()


def kill_procs():
    for j in range(len(__PROCS)):
        __WORK_QUEUE.put(None)


def _datacheck_peakdetect(x_axis, y_axis):
    if x_axis is None:
        x_axis = range(len(y_axis))

    if len(y_axis) != len(x_axis):
        raise ValueError(
               'Input vectors y_axis and x_axis must have same length')

    #needs to be a numpy array
    y_axis = np.asarray(y_axis)
    x_axis = np.asarray(x_axis)
    return x_axis, y_axis


def fit_quad_to_peak(x, y):
    """
    Fits a quadratic to the data points handed in
    to the from y = b[0](x-b[1]) + b[2]

    x -- locations
    y -- values

    returns (b, R2)

    """

    lenx = len(x)

    # some sanity checks
    if lenx < 3:
        raise Exception('insufficient points handed in ')
    # set up fitting array
    X = np.vstack((x ** 2, x, np.ones(lenx))).T
    # use linear least squares fitting
    beta, _, _, _ = np.linalg.lstsq(X, y)

    SSerr = np.sum(np.power(np.polyval(beta, x) - y, 2))
    SStot = np.sum(np.power(y - np.mean(y), 2))
    # re-map the returned value to match the form we want
    ret_beta = (beta[0],
                -beta[1] / (2 * beta[0]),
                beta[2] - beta[0] * (beta[1] / (2 * beta[0])) ** 2)

    return ret_beta, 1 - SSerr / SStot


def peakdetect(y_axis, x_axis=None, lookahead=300, delta=0):
    """
    Converted from/based on a MATLAB script at:
    http://billauer.co.il/peakdet.html

    function for detecting local maximas and minmias in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maximas and minimas respectively

    keyword arguments:
    y_axis -- A list containg the signal over which to find peaks
    x_axis -- (optional) A x-axis whose values correspond to the y_axis list
        and is used in the return to specify the postion of the peaks. If
        omitted an index of the y_axis is used. (default: None)
    lookahead -- (optional) distance to look ahead from a peak candidate to
        determine if it is the actual peak (default: 200)
        '(sample / period) / f' where '4 >= f >= 1.25' might be a good value
    delta -- (optional) this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
            delta function causes a 20% decrease in speed, when omitted
            Correctly used it can double the speed of the function

    return -- two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tupple
        of: (position, peak_value)
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do:
        x, y = zip(*tab)
    """
    max_peaks = []
    min_peaks = []
    dump = []   # Used to pop the first hit which almost always is false

    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = len(y_axis)

    #perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")

    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf

    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead],
                                       y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x

        ####look for max####
        if y < mx - delta and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index + lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index + lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
                continue
            #else:  #slows shit down this does
            #    mx = ahead
            #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]

        ####look for min####
        if y > mn + delta and mn != -np.Inf:
            #Minima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index + lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index + lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
            #else:  #slows shit down this does
            #    mn = ahead
            #    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]

    #Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        #no peaks were found, should the function return empty lists?
        pass

    return [max_peaks, min_peaks]


def peakdetect_parabole(y_axis, x_axis, min_pts=4, max_pts=25, R2_cut=.1, is_ring=False):
    """
    Function for detecting local maximas and minmias in a signal.
    Discovers peaks by fitting the model function: y = k (x - tau) ** 2 + m
    to the peaks.  The region between zero crossing is fit, the type of peak
    is determined by fit.

    keyword arguments:
    :param y_axis: A list containg the signal over which to find peaks
    :param x_axis:  A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the postion of the peaks.
    :param min_pts: the minimum points between a zero crossing for it to be peak candidate
    :param max_pts: regions with more points than this will be down sampled
    :param is_ring: if True, then data is periodic.

    return -- two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a list
        of: (position, peak_value)
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do:
        x, y = zip(*max_peaks)
    """
    if len(y_axis) != len(x_axis):
        raise ValueError('Input vectors y_axis and x_axis must have same length')

    # get zero crossings
    zero_indices = zero_crossings(y_axis) + 1
    period_lengths = np.diff(zero_indices)

    # define output variable
    max_peaks = []
    min_peaks = []

    work_list = []
    res_list = []
    for indx, p_len in zip(zero_indices, period_lengths):
        if p_len < min_pts:
            continue
        step = 1
        if p_len > max_pts:
            step = int(np.floor(p_len / max_pts))
        work_list.append((x_axis[(indx - 1):(indx + p_len + 1):step], y_axis[(indx - 1):(indx + p_len + 1):step]))

    # deal with ring
    if is_ring:
        if np.sign(y_axis[0]) == np.sign(y_axis[-2]):
            # if the first and last points have the same sign, we caught
            # all zero-crossings and just need to sticky tape the front
            # and back together
            tmp_x = np.concatenate((x_axis[zero_indices[-1] - 1:] - 2 * np.pi, x_axis[1:zero_indices[0] + 1]))
            tmp_y = np.concatenate((y_axis[zero_indices[-1] - 1:], y_axis[1:zero_indices[0] + 1]))
            p_len = len(tmp_x)
            if p_len >= min_pts:
                step = 1
                if p_len > max_pts:
                    step = int(np.floor(p_len / max_pts))
                work_list.append((tmp_x[::step], tmp_y[::step]))

        else:
            # there is a hidden zero crossing in the data
            # deal with first peak
            p_len = zero_indices[0] + 1
            if p_len >= min_pts:
                step = 1
                if p_len > max_pts:
                    step = int(np.floor(p_len / max_pts))
                work_list.append((x_axis[:zero_indices[0] + 1:step], y_axis[:zero_indices[0] + 1:step]))
            # deal with last
            p_len = len(y_axis) - zero_indices[-1]
            if p_len >= min_pts:
                step = 1
                if p_len > max_pts:
                    step = int(np.floor(p_len / max_pts))
                work_list.append((x_axis[zero_indices[-1]::step], y_axis[zero_indices[-1]::step]))

    if __PROCS is not None:
        # split up the work in a semi-equal way
        jobs_pre_proc = len(work_list) // len(__PROCS) + 1
        for j in range(len(__PROCS)):
            __WORK_QUEUE.put(work_list[j * jobs_pre_proc:(j + 1) * jobs_pre_proc])
        __WORK_QUEUE.join()

        # we know we never hand out more than len(__PROCS) jobs
        for j in range(len(__PROCS)):
            res_list.extend(__RES_QUEUE.get())
    else:
        res_list = [fit_quad_to_peak(x, y) for x, y in work_list]

    res_dict = {-1: [], 1: [], 0: []}

    for ((a, b, c), e), (x, y) in itertools.izip(res_list, work_list):
        if e > R2_cut:
            res_dict[np.sign(a)].append((b, c))
        else:

            orig_sign = np.sign(y[1])
            y = y * orig_sign
            step = len(y) // 4
            slice_arg = np.argmin(y[step:-step]) + step
            if slice_arg - 1 > min_pts:
                t_len = slice_arg - 1

                step = int(np.floor(t_len / max_pts)) if t_len > max_pts else 1
                slc = slice(0, slice_arg - 1, step)

                ((a, b, c), R2) = fit_quad_to_peak(x[slc], orig_sign * y[slc])


                if R2 > R2_cut:

                    res_dict[np.sign(a)].append((b, c))

            if len(y) - slice_arg - 1 > min_pts:
                t_len = slice_arg - 1
                step = int(np.floor(t_len / max_pts)) if t_len > max_pts else 1
                slc = slice(slice_arg + 1, -1, step)

                ((a, b, c), R2) = fit_quad_to_peak(x[slc], orig_sign * y[slc])


                if R2 > R2_cut:

                    res_dict[np.sign(a)].append((b, c))

    return [res_dict[-1], res_dict[1]]


def peakdetect_zero_crossing(y_axis, x_axis=None):
    """
    Function for detecting local maximas and minmias in a signal.
    Discovers peaks by dividing the signal into bins and retrieving the
    maximum and minimum value of each the even and odd bins respectively.
    Division into bins is performed by smoothing the curve and finding the
    zero crossings.

    Suitable for repeatable signals, where some noise is tolerated. Excecutes
    faster than 'peakdetect', although this function will break if the offset
    of the signal is too large. It should also be noted that the first and
    last peak will probably not be found, as this function only can find peaks
    between the first and last zero crossing.

    keyword arguments:
    y_axis -- A list containg the signal over which to find peaks
    x_axis -- (optional) A x-axis whose values correspond to the y_axis list
        and is used in the return to specify the postion of the peaks. If
        omitted an index of the y_axis is used. (default: None)
    window -- the dimension of the smoothing window; should be an odd integer
        (default: 11)

    return -- two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tupple
        of: (position, peak_value)
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do:
        x, y = zip(*tab)
    """
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)

    zero_indices = zero_crossings(y_axis)
    period_lengths = np.diff(zero_indices)

    bins_y = [y_axis[index:index + diff] for index, diff in
              zip(zero_indices, period_lengths)]
    bins_x = [x_axis[index:index + diff] for index, diff in
              zip(zero_indices, period_lengths)]

    even_bins_y = bins_y[::2]
    odd_bins_y = bins_y[1::2]
    even_bins_x = bins_x[::2]
    odd_bins_x = bins_x[1::2]
    hi_peaks_x = []
    lo_peaks_x = []

    min_peaks = []
    max_peaks = []

    # check if even bin contains maxima
    if abs(even_bins_y[0].max()) > abs(even_bins_y[0].min()):
        high_bins_y = even_bins_y
        high_bins_x = even_bins_x
        low_bins_y = odd_bins_y
        low_bins_x = odd_bins_x
    # or a minimum
    else:
        high_bins_y = odd_bins_y
        high_bins_x = odd_bins_x
        low_bins_y = even_bins_y
        low_bins_x = even_bins_x

    high_peaks = [_bin.argmax() for _bin in high_bins_y]
    low_peaks = [_bin.argmin() for _bin in low_bins_y]
    # get x values for peak
    for x, y, peak in zip(high_bins_x, high_bins_y, high_peaks):
        max_peaks.append((x[peak], y[peak]))
    for x, y, peak in zip(low_bins_x, low_bins_y, low_peaks):
        min_peaks.append((x[peak], y[peak]))
    return [max_peaks, min_peaks]


def zero_crossings(y_axis):
    """
    Algorithm to find zero crossings.  Finds the zero-crossings by
    looking for a sign change.  This assumes that the data has been
    sensibly smoothed before being handed in.


    keyword arguments:
    y_axis -- A list containg the signal over which to find zero-crossings

    return -- the index for each zero-crossing
    """

    zero_crossings, = np.diff(np.sign(y_axis)).nonzero()

    # check if any zero crossings were found
    if len(zero_crossings) < 1:
        raise ValueError("No zero crossings found")

    return zero_crossings
