#Copyright 2012 Thomas A Caswell
#tcaswell@uchicago.edu
#http://jfi.uchicago.edu/~tcaswell
#
#This program is free software; you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation; either version 3 of the License, or (at
#your option) any later version.
#
#This program is distributed in the hope that it will be useful, but
#WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program; if not, see <http://www.gnu.org/licenses>.
from __future__ import division

import hashlib
import time

import cine
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as sint
import scipy.odr as sodr
import numpy.linalg as nl
from trackpy.tracking import Point
from trackpy.tracking import Track

import h5py

WINDOW_DICT = {'flat':np.ones,'hanning':np.hanning,'hamming':np.hamming,'bartlett':np.bartlett,'blackman':np.blackman}


class hash_line_angular(object):
    '''1D hash table with linked ends for doing the ridge linking
    around a rim'''
    def __init__(self, dims, bin_width):
        '''The argument dims needs to be there to homogenize hash interfaces  '''
        full_width = 2*np.pi
        self.boxes = [[] for j in range(0, int(np.ceil(full_width/bin_width)))]
        self.bin_width = bin_width
        self.bin_count = len(self.boxes)
        
    def add_point(self, point):
        ''' Adds a point on the hash line

        Assumes that the point have been properly rationalized 0<phi<2pi
        '''
        self.boxes[int(np.floor(point.phi/self.bin_width))].append(point)

    def get_region(self, point, bbuffer):
        '''Gets the region around the point

        Assumes that the point have been properly rationalized 0<phi<2pi
        '''
        bbuffer = int(np.ceil(bbuffer/self.bin_width))

        box_indx = int(np.floor(point.phi/self.bin_width))
        tmp_box = []
        for j in range(box_indx - bbuffer, box_indx + bbuffer + 1):
            tmp_box.extend(self.boxes[np.mod(j, self.bin_count)])
        return tmp_box


class Point1D_circ(Point):
    '''
    Version of :py:class:`Point` for finding fringes

    :py:attr:`Point1D_circ.q` is the parameter for the curve where the point is (maps to time in standard tracking)
    :py:attr:`Point1D_circ.phi` is the angle of the point along the parametric curve
    :py:attr:`Point1D_circ.v` any extra values that the point should carry
    '''

    
    def __init__(self, q, phi, v=0):
        Point.__init__(self)                  # initialize base class
        self.q = q                            # parametric variable
        self.phi = np.mod(phi,2*np.pi)                        # 
        self.v = v                            # the value at the extrema (can probably drop this)

    def distance(self, point):
        '''Returns the absolute value of the angular distance between
        two points mod 2\pi'''
        d = np.abs(self.phi - point.phi)
        if d> np.pi:
            d = np.abs(2*np.pi - d)  
        return d

class lf_Track(Track):
    def __init__(self, point=None):
        Track.__init__(self, point)
        self.charge = None
        self.q = None
        self.phi = None

    def sort(self):
        self.points.sort(key = lambda x: x.q)

    def plot_trk(self, ax,**kwargs):
        if self.charge is None:
            kwargs['color'] = 'm'
        elif self.charge == 1:
            kwargs['color'] = 'r'
        elif self.charge == -1:
            kwargs['color'] = 'b'
        else:
            kwargs['color'] = 'c'

        ax.plot(*zip(*[(p.q,p.phi) for p in self.points]) ,**kwargs)
    def plot_trk_img(self,pram,ax,**kwargs):
        a,b,t0,x0,y0 = pram
        X,Y = np.hstack([gen_ellipse(a*p.q,b*p.q,t0,x0,y0,p.phi) for p in self.points])
        if self.charge is None:
            kwargs['marker'] = '*'
        elif self.charge == 1:
            kwargs['marker'] = '^'
        elif self.charge == -1:
            kwargs['marker'] = 'v'
        else:
            kwargs['marker'] = 'o'
        ax.plot(X,Y,**kwargs)
    def classify2(self,min_len = None,min_extent = None,**kwargs):
        ''' second attempt at the classify function''' 
        phi,q = zip(*[(p.phi,p.q) for p in self.points])
        q = np.asarray(q)
        # if the track is less than 25, don't try to classify
        
        if min_len is not None and len(phi) < min_len:
            self.charge =  None
            self.q = None
            self.phi = None
            return

        p_shift = 0
        if np.min(phi) < 0.1*np.pi or np.max(phi) > 2*np.pi*.9:
            p_shift = np.pi
            phi = np.mod(np.asarray(phi) + p_shift,2*np.pi)
            
        if min_extent is not None and np.max(phi) - np.min_phi   < min_extent:
            self.charge =  None
            self.q = None
            self.phi = None
            return

        a = np.vstack([q**2,q,np.ones(np.size(q))]).T
        X,res,rnk,s = nl.lstsq(a,phi)
        phif = a.dot(X)
        #        p = 1- ss.chi2.cdf(np.sum(((phif - phi)**2)/phif),len(q)-3)

        prop_c = -np.sign(X[0])
        prop_q = -X[1]/(2*X[0])
        prop_phi = prop_q **2 * X[0] + prop_q * X[1] + X[2]


        if prop_q < np.min(q) or prop_q > np.max(q):
            # the 'center' in outside of the data we have -> screwy track don't classify
            self.charge =  None
            self.q = None
            self.phi = None
            return

        self.charge = prop_c
        self.q = prop_q
        self.phi = prop_phi - p_shift
                        
    # classify tracks
    def classify(self):
        '''This needs to be re-written to deal with non-properly Chevron tracks better '''
        phi,a = zip(*[(p.phi,p.q) for p in self.points])
        self.phi = np.mean(phi)
        if len(phi) < 25:
            self.charge =  0
            self.q = 0
            return
        
        i_min = np.min(phi)
        i_max = np.max(phi)
        q_val = 0
        match_count = 0
        match_val = 0
        fliped = False
        while len(phi) >=15:
            # truncate the track
            phi = phi[4:-5]
            a = a[4:-5]
            # get the current min and max
            t_min = np.min(phi)
            t_max = np.max(phi)
            # if the min hasn't changed, claim track has negative charge
            if t_min == i_min:
                # if the track doesn't currently have negative charge
                if match_val != -1:
                    # if it currently has positive charge
                    if match_val == 1:
                        # keep track of the fact that it has flipped
                        fliped = True
                                    
                    match_val = -1        #change the proposed charge
                    q_val = a[np.argmin(phi)] #get the q val of the
                                              #minimum
                    match_count =0            #set the match count to 0
                match_count +=1               #if this isn't a change, increase the match count
            elif t_max == i_max:
                if match_val != 1:
                    if match_val == -1:
                        fliped = True
                        
                    match_val = 1
                    q_val = a[np.argmax(phi)]
                    
                    match_count =0
                match_count +=1
            elif t_max < i_max and match_val == 1: #we have truncated
                                                   #the maximum off at
                                                   #it is positively
                                                   #charged
                match_val = 0                      #reset
                mach_count = 0
                q_val = 0
                i_max = t_max
                i_min = t_min
                
            elif t_min > i_min  and match_val == -1: #we have truncated the minimum off at it is 
                match_val = 0                      #reset
                mach_count = 0
                q_val = 0
                i_max = t_max
                i_min = t_min
                
            if match_count == 2:          #if get two matches in a row
                self.charge = match_val
                if match_val == -1:
                    self.phi = i_min
                    self.q = q_val
                elif match_val == 1:
                    self.phi = i_max
                    self.q = q_val
                else:
                    print 'should not have hit here 1'
                return
        if not fliped:
            self.charge =  match_val
            if match_val == -1:
                self.phi = i_min
                self.q = q_val
            elif match_val == 1:
                self.phi = i_max
                self.q = q_val
            else:
                self.q = 0
#                print 'should not have hit here 2'
            return
        else:
            self.charge = 0
            self.q = 0
            return 
    def mean_phi(self):
        self.phi = np.mean([p.phi for p in self.points])
    def mean_q(self):
        self.q = np.mean([p.q for p in self.points])

    def merge_track(self,to_merge_track):
        pt.Track.merge_track(self,to_merge_track)
        if self.phi is not None:
            self.mean_phi()
        if self.charge is not None:
            self.classify()





def gen_ellipse(a,b,t,x,y,theta):
    # a is always the major axis, x is always the major axis, can be rotated away by t
    if b > a:
            tmp = b
            b = a
            a = tmp

            
    #t = np.mod(t,np.pi/2)
    r =  1/np.sqrt((np.cos(theta - t)**2 )/(a*a) +(np.sin(theta - t)**2 )/(b*b) )
    return np.vstack((r*np.cos(theta) + x,r*np.sin(theta) + y))

class ellipse_fitter(object):
    def __init__(self):
        self.pt_lst = []
        
        
    def click_event(self,event):
        ''' Extracts locations from the user'''
        if event.key == 'shift':
            self.pt_lst = []
            
        self.pt_lst.append((event.xdata,event.ydata))

    def get_params(self):
        return gen_to_parm(fit_ellipse(np.vstack(self.pt_lst).T).beta)

    def return_points(self):
        '''Returns the clicked points in the format the rest of the code expects'''
        return np.vstack(self.pt_lst).T

def hash_file(fname):
    """for computing hash values of files.  This is to make it easy to
   run my data base scheme with files that are on external hard drives.

   code lifted from:
   http://stackoverflow.com/a/4213255/380231
   """

    
    md5 = hashlib.md5()
    with open(fname,'rb') as f: 
        for chunk in iter(lambda: f.read(128*md5.block_size), b''): 
            md5.update(chunk)
    return md5.hexdigest()


def set_up_efitter(fname,bck_img = None):
    ''' gets the initial path '''
    clims = [.5,1.5]
    #open the first frame and find the initial circle
    c_test = cine.Cine(fname)    
    lfimg = c_test.get_frame(0)
    if bck_img is None:
        bck_img = np.ones(lfimg.shape)
        clims = None
    fig = plt.figure()
    ax = fig.add_axes([.1,.1,.8,.8])
    im = ax.imshow(lfimg/bck_img)
    if clims is not None:
        im.set_clim(clims)
    ef = ellipse_fitter()
    plt.connect('button_press_event',ef.click_event)


    plt.draw()

    return ef

def gen_bck_img(fname):
    '''Computes the background image'''
    c_test = cine.Cine(fname) 
    bck_img = reduce(lambda x,y:x+y,c_test,np.zeros(c_test.get_frame(0).shape))
    print c_test.len()
    bck_img/=c_test.len()
    # hack to deal with 
    bck_img[bck_img==0] = .001
    return bck_img


def disp_frame(fname,n,bck_img = None):
    '''Displays a given frame from the file'''

    c_test = cine.Cine(fname)    

    lfimg = c_test.get_frame(n)

    if bck_img is None:
        bck_img = np.ones(lfimg.shape)
    fig = plt.figure()
    ax = fig.add_axes([.1,.1,.8,.8])
    im = ax.imshow(lfimg/bck_img)
    im.set_clim([.5,1.5])
    ax.set_title(n)
    plt.draw()


def play_movie(fname,bck_img=None):
    '''plays the movie with correction'''
    

    def update_img(num,F,bck_img,im,txt):
        im.set_data(F.get_frame(num)/bck_img)
        txt.set_text(str(num))
    F = cine.Cine(fname)    
    
    if bck_img is None:
        bck_img = np.ones(F.get_frame(0).shape)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    im = ax.imshow(F.get_frame(0)/bck_img)
    fr_num = ax.text(0.05,0.05,0,transform = ax.transAxes )
    im.set_clim([.75,1.25])
    prof_ani = animation.FuncAnimation(fig,update_img,len(F),fargs=(F,bck_img,im,fr_num),interval=50)
    plt.show()


def plot_plst_data(p_lst):
    ''' makes a graph for the position, radius, angle, etc from the list of ellipse parameters'''

    

    print 'hi'
    a,b,t0,x0,y0 = zip(*p_lst)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(a,label='a')
    ax.plot(b,label='b')
    ax.set_ylabel('axis [pix]')
    ax.set_xlabel(r'frame \#')



    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x0,y0)
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')
    ax.set_aspect('equal')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x0,label='x0')
    ax.plot(y0,label='y0')
    ax.legend(loc=0)
    ax.set_ylabel('center location [px]')
    ax.set_xlabel('frame \#')
    
    x0 = np.array(x0)
    y0 = np.array(y0)
    x0_0 = x0-np.mean(x0)
    y0_0 = y0-np.mean(y0)
    x0_0 = x0_0/np.sqrt(np.sum(x0_0**2))
    y0_0 = y0_0/np.sqrt(np.sum(y0_0**2))
    
    print sum(x0_0 * y0_0)
    



    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.step(range(len(t0)),t0,label=r'$\theta_0$')
    ax.axhline(-np.pi/2)
    ax.axhline(np.pi/2)
    ax.axhline(-np.pi)
    ax.axhline(np.pi)
    ax.legend(loc=0)
    ax.set_ylabel(r'$\theta_0$ [rad]')
    ax.set_xlabel('frame \#')
    


    
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(10*(np.array(t0) - np.mean(t0)),label=r'$\theta_0$')
    ax.plot(a-np.mean(a),label='a')
    ax.plot(b-np.mean(b),label='b')
    ax.set_ylabel('axis [pix]')
    ax.set_xlabel('frame \#')
    ax.legend(loc=0)
    ax.set_ylabel(r'arb')
    ax.set_xlabel('frame \#')
    



def resample_track(data,pt_num = 250,interp_type = 'linear'):
    '''re-samples the curve on uniform points and averages out tilt
    due to fringe ID error'''

    # get data out
    ch,th = data
    th = np.array(th)
    ch = np.array(ch)
    
    # make negative points positive
    th = np.mod(th,2*np.pi)
    indx = th.argsort()
    # re-order to be monotonic
    th = th[indx]
    ch = ch[indx]
    # sum the charges
    ch = np.cumsum(ch)

    # figure out the miss/match
    miss_cnt = ch[-1]
    corr_ln =th*(miss_cnt/(2*np.pi)) 
    # add a linear line to make it come back to 0
    ch -= corr_ln

    # make sure that the full range is covered
    if th[0] != 0:
        ch = np.concatenate((ch[:1],ch))
        th = np.concatenate(([0],th))
    if th[-1] < 2*np.pi:
        ch = np.concatenate((ch,ch[:1]))
        th = np.concatenate((th,[2*np.pi]))

    # set up interpolation 
    f = sint.interp1d(th,ch,kind=interp_type)
    # set up new points
    th_new = np.linspace(0,2*np.pi,pt_num)
    # get new interpolated values
    ch_new = f(th_new)
    # subtract off mean
    ch_new -=np.mean(ch_new)
    return ch_new,th_new


def e_funx(p,r):
    x,y = r
    a,b,c,d,f = p
        
    return a* x*x + 2*b*x*y + c * y*y + 2 *d *x + 2 * f *y -1

def fit_ellipse(r):
    x,y = r
    R2 = np.max(x - np.mean(x))**2

    a = c = 1/R2
    b = 0
    d = -np.mean(x)/R2
    f = -np.mean(y)/R2
    
    p0 = (a,a,c,d,f)
    data = sodr.Data(r,1)
    model = sodr.Model(e_funx,implicit=1)
    worker = sodr.ODR(data,model,p0)
    out = worker.run()
    out = worker.restart()
    return out

# http://mathworld.wolfram.com/Ellipse.html
def gen_to_parm(p):
    a,b,c,d,f = p
    g = -1
    x0 = (c*d-b*f)/(b*b - a*c)
    y0 = (a*f - b*d)/(b*b - a*c)
    ap = np.sqrt((2*(a*f*f + c*d*d + g*b*b - 2*b*d*f - a*c*g))/((b*b - a*c) * (np.sqrt((a-c)**2 + 4 *b*b)-(a+c))))
    bp = np.sqrt((2*(a*f*f + c*d*d + g*b*b - 2*b*d*f - a*c*g))/((b*b - a*c) * (-np.sqrt((a-c)**2 + 4 *b*b)-(a+c))))

    t0 =  (1/2) * np.arctan(2*b/(a-c))
    
    if a>c: 
        t0 =  (1/2) * np.arctan(2*b/(a-c))
        
    else:
        t0 = np.pi/2 + (1/2) * np.arctan(2*b/(c-a))
        
    

    return (ap,bp,t0,x0,y0)


def l_smooth(values,window_len=2,window='flat'):
    window_len = window_len*2+1
    s=np.r_[values[window_len-1:0:-1],values,values[-1:-window_len:-1]]
    w = WINDOW_DICT[window](window_len)
    #    w = np.ones(window_len,'d')
    #w = np.exp(-((np.linspace(-(window_len//2),window_len//2,window_len)/(window_len//4))**2)/2)
    
    values = np.convolve(w/w.sum(),s,mode='valid')[(window_len//2):-(window_len//2)]
    return values


def do_comp():
    pass

def save_comp(fout_base,fout_path,fout_name,params):
    # check status of h5 file

    # either open or create h5 file

    pass

def _write_frame_tracks_to_file(parent_group,t_min_lst,t_max_lst,md_args):
    ''' 
    Takes in an hdf object and creates the following data sets in `parent_group`


    raw_data_{min,max}
        a 2xN array with columns {ma,phi} which
        is all of the tracked points for this frame

    raw_track_md_{min,max}
        a 2x(track_count) array with columns
        {track_len,start_index} Start index refers to
        raw_data_{}
    
    trk_res_{min,max}
        a 2x(track_count) array with columns {charge,phi}

    everything in md_args is shoved into the group level attributes

    ======
    INPUT
    ======
    `parent_group`
        h5py group object.  Should not contain existing data sets with the same names

    `t_min_lst`
        an iterable of the tracks for the minimums in the frame.  

    `t_max_lst`
        an iterable of the tracks for the minimums in the frame

    `md_args`
        a dictionary of meta-data to be attached to the group
    '''

    # names
    raw_data_name = 'raw_data_'
    raw_track_md_name = 'raw_track_md_'
    trk_res_name = 'trk_res_'
    name_mod = ('min','max')
    write_raw_data = True
    write_res = True
    for t_lst,n_mod in zip((t_min_lst,t_max_lst),name_mod):
        if write_raw_data:
            # get total number of points
            pt_count = np.sum([len(t) for t in t_lst])
            # arrays to accumulate data into
            tmp_raw_data = np.zeros((pt_count,2))
            tmp_raw_track_data = np.zeros((len(t_lst),2))
            tmp_indx = 0
            print pt_count
            for i,t in enumerate(t_lst):
                t_len = len(t)
                # shove in raw data
                tmp_raw_data[tmp_indx:(tmp_indx + t_len), 0] = np.array([p.q for p in t])
                tmp_raw_data[tmp_indx:(tmp_indx + t_len), 1] = np.array([p.phi for p in t])
                # shove in raw track data
                tmp_raw_track_data[i,:] = (t_len,tmp_indx)
                # increment index
                tmp_indx += t_len
          
            # create dataset and shove in data
            parent_group.create_dataset(raw_data_name + n_mod,tmp_raw_data.shape,np.float,compression='szip')
            parent_group[raw_data_name + n_mod][:] = tmp_raw_data

            parent_group.create_dataset(raw_track_md_name + n_mod,tmp_raw_track_data.shape,np.float,compression='szip')
            parent_group[raw_track_md_name + n_mod][:] = tmp_raw_track_data
            
        if write_res:
            good_t_lst  = [t for t in t_lst if t.charge is not None or t.charge != 0]
            tmp_track_res = np.zeros((len(good_t_lst),3))

            # shove in results data
            for i,t in enumerate(good_t_lst):
                tmp_track_res[i,:] = (t.charge,t.phi,t.q)
                
            parent_group.create_dataset(trk_res_name + n_mod,tmp_track_res.shape,np.float,compression='szip')
            parent_group[trk_res_name + n_mod][:] = tmp_track_res
            
    for key,val in md_args.iteritems():
        g.attrs[key] = val


def _read_frame_tracks_from_file_raw(parent_group):
    '''
    inverse operation to `_write_frame_tracks_to_file`

    Reads out all of the raw data

    '''
    
    # names
    raw_data_name = 'raw_data_'
    raw_track_md_name = 'raw_track_md_'
    name_mod = ('min','max')
    
    trk_lsts_tmp = []
    for n_mod in name_mod:
        tmp_raw_data = parent_group[raw_data_name + n_mod][:]
        tmp_track_data = parent_group[raw_track_md_name + n_mod][:]
        t_lst = []
        for t_len,strt_indx in tmp_track_data:
            tmp_trk = lf_Track()
            for ma,phi in tmp_raw_data[strt_indx:(strt_indx + t_len),:]:
                tmp_trk.add_point(Point1D_circ(ma,phi)) 
            tmp_trk.classify2()
            t_lst.append(tmp_trk)
        trk_lsts_tmp.append(t_lst)

    return trk_lsts_tmp



def _read_frame_tracks_from_file_res(parent_group):
    '''
    Only reads out the charge and location of the tracks, not all of their points
    '''
    
    
    # names
    trk_res_name = 'trk_res_'
    name_mod = ('min','max')
    res_lst = []
    for n_mod in name_mod:
        tmp_trk_res = parent_group[trk_res_name + n_mod][:]
        tmp_charge = tmp_trk_res[:,0]
        tmp_phi = tmp_trk_res[:,1]
        tmp_q = tmp_trk_res[:,2]
        res_lst.appendf((tmp_charge,tmp_phi,tmp_q))

    return res_lst
