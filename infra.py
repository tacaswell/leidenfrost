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
import collections
import warnings
import os

import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage
import scipy.interpolate as sint
import scipy.interpolate as si
import scipy.odr as sodr

import h5py
import cine
from trackpy.tracking import Point
from trackpy.tracking import Track
import find_peaks.peakdetect as pd
import trackpy.tracking as pt




FilePath = collections.namedtuple('FilePath',['base_path','path','fname'])

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
    def plot_trk_img(self,tck,center,ax,**kwargs):
        if len(self.points) <2:
            return
        new_pts = si.splev(np.array([np.mod(p.phi,2*np.pi)/(2*np.pi) for p in self.points]),tck)
        

        new_pts -= center

        th = np.arctan2(*(new_pts[::-1]))

        # compute radius
        r = np.sqrt(np.sum(new_pts**2,axis=0))

        r *= np.array([p.q for p in self.points])

        zp_all = np.vstack(((np.cos(th)*r),(np.sin(th)*r))) + center
        
        if self.charge is None:
            kwargs['marker'] = '*'
        elif self.charge == 1:
            kwargs['marker'] = '^'
        elif self.charge == -1:
            kwargs['marker'] = 'v'
        else:
            kwargs['marker'] = 'o'
        if 'markevery' not in kwargs:
            kwargs['markevery'] = 10
        if 'markersize' not in kwargs:
            kwargs['markersize'] = 7.5
            
        ax.plot(*zp_all,**kwargs)
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
        
        # if the track does not straddle the seed curve, probably junk
        if np.min(q) > 1 or np.max(q) < 1:
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






def get_spline(points,point_count=None,pix_err = 2,**kwargs):
    '''
    Returns a closed spline for the points handed in.  Input is assumed to be a (2xN) array

    =====
    input
    =====

    points
        a 2xN array 

    point_count (optional)
        the number of new places to sample
        
    center
        The center of the point for converting to a shifted radial coordinate system
    =====
    output
    =====
    new_points
        a 2x{N,point_count} array with evenly sampled points
    tck
       The return data from the spline fitting
    center
       The center of mass the initial points
    '''

    if type(points) is np.ndarray:
        # make into a list
        pt_lst = zip(*points)
        # get center
        center = np.mean(points,axis=1).reshape(2,1)
    else:
        # make a copy of the list
        pt_lst = list(points)
        # compute center
        center = np.array(reduce(lambda x,y: (x[0] + y[0],x[1] + y[1]),pt_lst)).reshape(2,1)/len(pt_lst)

    if len(pt_lst)<5:
        raise Exception("not enough points")


    # sort the list by angle around center
    pt_lst.sort(key=lambda x: np.arctan2(x[1]-center[1],x[0]-center[0]))
    # add first point to end because it is periodic (makes the interpolation code happy)
    pt_lst.append(pt_lst[0])

    # make array for handing in to spline fitting
    pt_array = np.vstack(pt_lst).T
    # do spline fitting

    tck,u = si.splprep(pt_array,s=len(pt_lst)*(pix_err**2),per=True)
    if point_count is not None:
        new_pts = si.splev(np.linspace(0,1,point_count),tck)
        center = np.mean(points,axis=1).reshape(2,1)
    else:
        new_pts = si.splev(np.linspace(0,1,1000),tck)
        center = np.mean(points,axis=1).reshape(2,1)
        new_pts = []
    pt_lst.pop(-1)
    return new_pts,tck,center



    

class spline_fitter(object):
    def __init__(self,ax,pix_err = 1):
        fig = ax.get_figure()
        fig.canvas.mpl_connect('button_press_event',self.click_event)
        self.pt_lst = []
        self.pt_plot = ax.plot([],[],marker='x',linestyle ='-')[0]
        self.sp_plot = ax.plot([],[],lw=3,color='k')[0]
        self.pix_err = pix_err
        
    def click_event(self,event):
        ''' Extracts locations from the user'''
        if event.key == 'shift':
            self.pt_lst = []
            return
        if event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            self.pt_lst.append((event.xdata,event.ydata))
        elif event.button == 3:
            self.remove_pt((event.xdata,event.ydata))
        
        self.redraw()

    def remove_pt(self,loc):
        self.pt_lst.pop(np.argmin(map(lambda x:np.sqrt( (x[0] - loc[0])**2 + (x[1] - loc[1])**2),self.pt_lst)))
    def redraw(self):
        if len(self.pt_lst) > 5:
            new_pts,tck,center = get_spline(self.pt_lst,point_count=1000,pix_err = self.pix_err)
            self.sp_plot.set_xdata(new_pts[0])
            self.sp_plot.set_ydata(new_pts[1])
            self.pt_lst.sort(key=lambda x: np.arctan2(x[1]-center[1],x[0]-center[0]))
        else:
            self.sp_plot.set_xdata([])
            self.sp_plot.set_ydata([])
        x,y = zip(*self.pt_lst)
        self.pt_plot.set_xdata(x)
        self.pt_plot.set_ydata(y)

        plt.draw()


    def get_params(self):
        return gen_to_parm(fit_ellipse(np.vstack(self.pt_lst).T).beta)

    def return_points(self):
        '''Returns the clicked points in the format the rest of the code expects'''
        return np.vstack(self.pt_lst).T




def gen_bck_img(fname):
    '''Computes the background image'''
    c_test = cine.Cine(fname) 
    bck_img = reduce(lambda x,y:x+y,c_test,np.zeros(c_test.get_frame(0).shape))
    print c_test.len()
    bck_img/=c_test.len()
    # hack to deal with 
    bck_img[bck_img==0] = .001
    return bck_img


WINDOW_DICT = {'flat':np.ones,'hanning':np.hanning,'hamming':np.hamming,'bartlett':np.bartlett,'blackman':np.blackman}
def l_smooth(values,window_len=2,window='flat'):
    window_len = window_len*2+1
    s=np.r_[values[-(window_len-1):],values,values[0:(window_len-1)]]
    w = WINDOW_DICT[window](window_len)
    #    w = np.ones(window_len,'d')
    #w = np.exp(-((np.linspace(-(window_len//2),window_len//2,window_len)/(window_len//4))**2)/2)
    
    values = np.convolve(w/w.sum(),s,mode='valid')[(window_len//2):-(window_len//2)]
    return values




def _write_frame_tracks_to_file(parent_group,t_min_lst,t_max_lst,curve,md_args={}):
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
    curve.write_to_hdf(parent_group)
    for key,val in md_args.items():
        try:
            parent_group.attrs[key] = val
        except TypeError:
            print 'key: ' + key + ' can not be gracefully shoved into an hdf object, ' 
            print 'please reconsider your life choices'
    for t_lst,n_mod in zip((t_min_lst,t_max_lst),name_mod):
        if write_raw_data:
            # get total number of points
            pt_count = np.sum([len(t) for t in t_lst])
            # arrays to accumulate data into
            tmp_raw_data = np.zeros((pt_count,2))
            tmp_raw_track_data = np.zeros((len(t_lst),2))
            tmp_indx = 0
            #            print pt_count
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
            parent_group.create_dataset(raw_data_name + n_mod,
                                        tmp_raw_data.shape,
                                        np.float,
                                        compression='szip')
            parent_group[raw_data_name + n_mod][:] = tmp_raw_data

            parent_group.create_dataset(raw_track_md_name + n_mod,
                                        tmp_raw_track_data.shape,
                                        np.float,
                                        compression='szip')
            parent_group[raw_track_md_name + n_mod][:] = tmp_raw_track_data
            
        if write_res:
            good_t_lst  = [t for t in t_lst if t.charge is not None and t.charge != 0]
            tmp_track_res = np.zeros((len(good_t_lst),3))

            # shove in results data
            for i,t in enumerate(good_t_lst):
                tmp_track_res[i,:] = (t.charge,t.phi,t.q)
                
            parent_group.create_dataset(trk_res_name + n_mod,
                                        tmp_track_res.shape,
                                        np.float,
                                        compression='szip')
            parent_group[trk_res_name + n_mod][:] = tmp_track_res
            

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
    center = parent_group.attrs['center']
    tck = [parent_group.attrs['tck0'],parent_group.attrs['tck1'],parent_group.attrs['tck2']]
    
    # names
    trk_res_name = 'trk_res_'
    name_mod = ('min','max')
    res_lst = []
    for n_mod in name_mod:
        tmp_trk_res = parent_group[trk_res_name + n_mod][:]
        tmp_charge = tmp_trk_res[:,0]
        tmp_phi = tmp_trk_res[:,1]
        tmp_q = tmp_trk_res[:,2]
        res_lst.append((tmp_charge,tmp_phi,tmp_q))

    return res_lst




def gen_stub_h5(cine_fname,h5_fname,params,seed_curve,bck_img = None):
    for s in ProcessBackend.req_args_lst:
        if s not in params:
            raise RuntimeError('Necessary key ' + s + ' not included')
    proc_path = '/'.join(h5_fname[:2])
    print proc_path
    if not os.path.exists(proc_path):
        os.makedirs(proc_path,0751)        

    file_out = h5py.File('/'.join(h5_fname),'w-')
    
    file_out.attrs['ver'] = '0.1.1'
    for key,val in params.items():
        try:
            file_out.attrs[key] = val
        except TypeError:
            print 'key: ' + key + ' can not be gracefully shoved into an hdf object, ' 
            print 'please reconsider your life choices'
        except Exception as e:
            print "FAILURE WITH HDF: " + e.__str__()
               
    file_out.attrs['cine_path'] = cine_fname.path
    file_out.attrs['cine_fname'] = cine_fname.fname

    if seed_curve is not None:
        seed_curve.write_to_hdf(file_out)
    if bck_img is not None:
        file_out.create_dataset('bck_img',
                                bck_img.shape,
                                np.float,
                                compression='szip')
        file_out['bck_img'][:] = bck_img
        
    file_out.close()
    
    

class ProcessBackend(object):
    req_args_lst = ['search_range','s_width','s_num','pix_err']
    def __len__(self):
        if self.cine_ is not None:
            return len(self.cine_)
        else:
            return 0
    def __init__(self):
        self.frames = []                  # list of the frames processed
        
        self.params = {}                  # the parameters to feed to proc_frame

        self.cine_fname = None   # file name
        self.cine_ = None                 # the cine object

        self.bck_img = None              # back ground image for normalization

        
    @classmethod
    def from_hdf_file(cls,cine_base_path,h5_fname):
        ''' Sets up object to process data based on MD in an hdf file.
        '''
        self = cls()
        tmp_file = h5py.File('/'.join(h5_fname),'r')
        keys_lst = tmp_file.attrs.keys()
        lc_req_args = ['tck0','tck1','tck2','center']
        h5_req_args = ['cine_path','cine_fname']
        for s in cls.req_args_lst + lc_req_args + h5_req_args:
            if s not in keys_lst:
                tmp_file.close()
                raise Exception("missing required argument %s"%s)
        
        self.params = dict(tmp_file.attrs)

        for k in lc_req_args:
            del self.params[k]

        self.cine_fname = FilePath(cine_base_path,self.params.pop('cine_path'),self.params.pop('cine_fname'))
        self.cine_ = cine.Cine('/'.join(self.cine_fname))


        if 'bck_img' in tmp_file.keys():
            self.bck_img = tmp_file['bck_img'][:]
        else:
            self.bck_img = gen_bck_img('/'.join(self.cine_fname))

        seed_curve = SplineCurve.from_hdf(tmp_file)
        
        tmp_file.close()
        
        return self,seed_curve

    @classmethod
    def from_args(cls,cine_fname,h5_fname=None,*args,**kwargs):
        self = cls()
        '''Sets up the object based on arguments
        '''

        for s in cls.req_args_lst:
            if s not in kwargs:
                raise Exception("missing required argument %s"%s)

        self.params = kwargs
        try:
            self.bck_img = self.params.pop('bck_img')
        except KeyError:
            self.bck_img = None
                    
        self.cine_fname = cine_fname
                
        self.cine_ = cine.Cine('/'.join(self.cine_fname))

        if self.bck_img is None:
            self.bck_img = gen_bck_img('/'.join(self.cine_fname))
        


        
        return self
            
    def process_frame(self,frame_number,curve):


        # get the raw data, and convert to float
        tmp_img = np.array(self.cine_.get_frame(frame_number),dtype='float')
        # if 
        if self.bck_img is not None:
            tmp_img /= self.bck_img
        tm,trk_res,tim,tam,miv,mav = proc_frame(curve,tmp_img,**self.params)
        
        mbe = MemBackendFrame(curve,frame_number,res = trk_res,trk_lst = [tim,tam],img = tmp_img)
        mbe.tm = tm

        return mbe,mbe.get_next_spline(**self.params)

            
class MemBackendFrame(object):
    """A class for keeping all of the relevant results about a frame in memory

    This class will get smarter over time.  

     - add logic to generate res from raw
     - add visualization code to this object
    """
    def __init__(self,curve,frame_number,res,trk_lst=None,img = None,*args,**kwarg):
        self.curve = curve
        self.res = res
        self.trk_lst =trk_lst
        self.frame_number = frame_number
        self.next_curve = None
        self.img = img
        self.mix_in_count = None
        new_res = []
        for t_ in self.res:
            tmp = ~np.isnan(t_[0])
            tmp_lst = [np.array(r)[tmp] for r in t_]
            new_res.append(tuple(tmp_lst))
        self.res = new_res
        
        pass
    def get_next_spline(self,mix_in_count=0,**kwargs):
        if self.next_curve is not None and self.mix_in_count == mix_in_count:
            return self.next_curve


        tim,tam = self.trk_lst

        # this is a parameter to forcibly mix in some number of points from the last curve

        t_q = np.array([t.q for t in tim+tam if 
                        t.q is not None  
                        and t.phi is not None 
                        and t.charge is not None
                        and t.charge != 0] + 
                        [1]*mix_in_count)

        t_phi = np.array([np.mod(t.phi,(2*np.pi))/(2*np.pi) for t in tim+tam if 
                        t.q is not None  
                        and t.phi is not None 
                        and t.charge is not None
                        and t.charge != 0] +  
                        list(np.linspace(0,1,mix_in_count,endpoint=False)))

        # seed the next round of points

        # get the (r,t) of _this_ frames spline
        r,th = self.curve.sample_rt(t_phi)
        # scale the radius
        r *= t_q
        # sort by theta
        indx =th.argsort()
        r = r[indx]
        th = th[indx]
        # generate the new curve

        new_curve = SplineCurve.from_pts(self.curve.rt_to_xy(r,th),**kwargs)

        self.next_curve = new_curve
        self.mix_in_count = mix_in_count
        
        return new_curve

    def plot_tracks(self,min_len = 0):
        fig = plt.figure();
        ax = fig.gca()
        if self.img is not None:
            c_img = ax.imshow(self.img,cmap=plt.get_cmap('gray'),interpolation='nearest');
            c_img.set_clim([.5,1.5])
        color_cycle = ['r','b']
        for tk_l,c in zip(self.trk_lst,color_cycle):
            [t.plot_trk_img(self.curve.tck,self.curve.center,ax,color=c,linestyle='-') for t in tk_l if len(t) > min_len ];
        ax.plot(*self.curve.get_xy_samples(1000))
        plt.draw();

    def write_to_hdf(self,parent_group):
        print 'frame_%05d'%self.frame_number
        group = parent_group.create_group('frame_%05d'%self.frame_number)
        _write_frame_tracks_to_file(group,self.trk_lst[0],self.trk_lst[1],self.curve)
        del group

    
class HdfBackend(object):
    """A class that wraps around an HDF results file"""
    def __init__(self,fname,cine_base_path = None,*args,**kwargs):
        self.file = h5py.File('/'.join(fname),'r')
        self.raw = True
        self.res = True
        if 'bck_img' in self.file.keys():
            self.bck_img = self.file['bck_img'][:]
        else:
            self.bck_img = None
        if cine_base_path is not None:
            self.cine_fname = cine_base_path + '/' + self.file.attrs['cine_path'] + '/' + self.file.attrs['cine_fname']
            self.cine = cine.Cine(self.cine_fname)
        else:
            self.cine_fname = None
            self.cine = None
        self.frames = {}
        pass

    def __del__(self):
        self.file.close()
    def get_frame(self,frame_num,*args,**kwargs):
        if frame_num not in self.frames:
            trk_lst = None
            res = None
            img = None
            g = self.file['frame_%05d'%frame_num]
            if self.raw:
                trk_lst = _read_frame_tracks_from_file_raw(g)
            if self.res:
                res = _read_frame_tracks_from_file_res(g)
            curve = SplineCurve.from_hdf(g)
            if self.cine is not None:
                img = np.array(self.cine.get_frame(frame_num),dtype='float')
                if self.bck_img is not None:
                    img /= self.bck_img
            self.frames[frame_num] = MemBackendFrame(curve,frame_num,res,trk_lst,img=img)
        return self.frames[frame_num]
    def gen_back_img(self):
        if self.cine_fname is not None:
            self.bck_img = gen_bck_img(self.cine_fname)
    

            

class SplineCurve(object):
    '''
    A class that wraps the scipy.interpolation objects
    '''
    @classmethod
    def from_pts(cls,new_pts,**kwargs):
        _,tck,center = get_spline(new_pts,**kwargs)
        this = cls(tck,center)
        this.raw_pts = new_pts
        return this

    @classmethod
    def from_hdf(cls,parent_group):
        center = parent_group.attrs['center']
        tck = [parent_group.attrs['tck0'],parent_group.attrs['tck1'],parent_group.attrs['tck2']]
        return cls(tck,center)
    
    def __init__(self,tck,center):
        '''A really hacky way of doing different 
        '''
        self.tck = tck
        self.center = center
    def get_xy_samples(self,sample_count):
        '''
        Returns the x-y coordinates of uniformly sampled points on the
        spline.  
        '''
        return si.splev(np.linspace(0,1,sample_count),self.tck)
    def get_rt_samples(self,sample_count):
        '''

        '''
        new_pts = self.get_xy_samples(sample_count)
        new_pts -= self.center
        return np.sqrt(np.sum(new_pts**2,axis=0)).reshape(1,-1),np.arctan2(*(new_pts[::-1]))

    def write_to_hdf(self,parent_group):
        parent_group.attrs['tck0'] = self.tck[0]
        parent_group.attrs['tck1'] = np.vstack(self.tck[1])
        parent_group.attrs['tck2'] = self.tck[2]
        parent_group.attrs['center'] = self.center

    def circumference(self):
        '''returns a rough estimate of the circumference'''
        new_pts = self.get_xy_samples(100)
        return np.sum(np.sqrt(np.sum(np.diff(new_pts,axis=1)**2,axis=0)))

    def sample_rt(self,points):
        '''Samples at the given points and returns the locations in (r,t)'''
        tmp_pts = si.splev(points,self.tck)
        tmp_pts -= self.center
        th = np.arctan2(*(tmp_pts[::-1]))
        r = np.sqrt(np.sum(tmp_pts**2,axis=0))

        return r,th

    def rt_to_xy(self,r,th):
        '''converts (r,t) coords to (x,y)'''
        return np.vstack(((np.cos(th)*r),(np.sin(th)*r))) + self.center
        

        

        
def find_rim_fringes(curve, lfimg, s_width, s_num, smooth_rng=2, *args, **kwargs):
    """
    Does the actual work of finding the fringes on the image
    """

    # a really rough estimate of the circumference 
    C = curve.circumference()

    # sample points at ~ 2/pix
    sample_count = int(np.ceil(C*2))

    r_new,th_new = curve.get_rt_samples(sample_count)


    # get center of curve
    x0,y0 = curve.center[:,0]

    
    R = np.max(r_new)*(1+s_width)*1.1
    x_shift = int(x0-R)
    if x_shift<0:
        x_shift = 0
    x_lim = int(x0+R)
    y_shift = int(y0-R)
    if y_shift < 0:
        y_shift = 0
    y_lim = int(y0+R)
    dlfimg = lfimg[y_shift:y_lim,x_shift:x_lim]

    # this will approximately  double sample.
    ma_scale_vec = np.linspace(1-s_width,1 +s_width,s_num).reshape(-1,1)

    
    r_scaled = ma_scale_vec.dot(r_new)

    X = np.cos(th_new)*r_scaled
    Y = np.sin(th_new)*r_scaled
    zp_all = np.vstack(((Y).reshape(-1),(X).reshape(-1))) + np.flipud(curve.center) - np.array((y_shift,x_shift)).reshape(2,1)

    # extract the values at those locations from the image.  The
    # extra flipud is to take a transpose of the points to deal
    # with the fact that the definition of the first direction
    # between plotting and the image libraries is inconsistent.
    zv_all = scipy.ndimage.interpolation.map_coordinates(dlfimg,zp_all,order=2)
    min_vec = []
    max_vec = []
    theta = np.linspace(0,2*np.pi,sample_count)
    for j,ma_scale in enumerate(ma_scale_vec.reshape(-1)):

        # select out the right region
        zv = zv_all[j*sample_count:(j+1)*sample_count] 
        # smooth the curve
        zv = l_smooth(zv,smooth_rng,'blackman')

        # find the peaks
        peaks = pd.peakdetect_parabole(zv-np.mean(zv),theta,is_ring =True)
        # extract the maximums
        max_pk = np.vstack(peaks[0])
        # extract the minimums
        min_pk = np.vstack(peaks[1])
        
        # append to the export vectors
        min_vec.append((ma_scale,min_pk))
        max_vec.append((ma_scale,max_pk))
        
        
    return min_vec,max_vec


def proc_frame(curve,img,s_width,s_num,search_range,min_tlen = 5, **kwargs):
    '''new version with different returns'''
    _t0 = time.time()


    miv,mav = find_rim_fringes(curve,img,s_width=s_width,s_num=s_num,**kwargs)

    tim = link_ridges(miv,search_range,**kwargs)
    tam = link_ridges(mav,search_range,**kwargs)

    tim = [t for t in tim if len(t) > min_tlen]
    tam = [t for t in tam if len(t) > min_tlen]

    trk_res = (zip(*[ (t.charge,t.phi) for t in tim if t.charge is not None ]),zip(*[ (t.charge,t.phi) for t in tam if t.charge is not None ]))


    _t1 = time.time()

    return (_t1 - _t0),trk_res,tim,tam,miv,mav
    
def link_ridges(vec,search_range,memory=0,**kwargs):
    # generate point levels from the previous steps

    levels = [[Point1D_circ(q,phi,v) for phi,v in pks] for q,pks in vec]
    
    trks = pt.link_full(levels,2*np.pi,search_range,hash_cls = hash_line_angular,memory = memory, track_cls = lf_Track)        
    for t in trks:
        t.classify2(**kwargs)

    trks.sort(key=lambda x: x.phi)
    return trks
