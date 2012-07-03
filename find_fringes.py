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


import numpy as np
import trackpy.tracking as pt
import scipy
import scipy.ndimage
import infra
import find_peaks as fp
import numpy.linalg as nl
import scipy.stats as ss
import time
from trackpy.tracking import Point
from trackpy.tracking import Track


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
        ''' Adds a point on the hash line'''
        t = np.mod(point.phi, 2*np.pi)
        self.boxes[int(np.floor(t/self.bin_width))].append(point)

    def get_region(self, point, bbuffer = 1):
        '''Gets the region around the point'''
        bbuffer = int(np.ceil(bbuffer/self.bin_width))
        box_indx = int(np.floor(self.bin_count * np.mod(point.phi, 2*np.pi)/(2*np.pi)))
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

    
    def __init__(self, q, phi, v):
        Point.__init__(self)                  # initialize base class
        self.q = q                            # parametric variable
        self.phi = phi                        # 
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
        ax.plot(*zip(*[(p.q,p.phi) for p in self.points]) ,**kwargs)
    def plot_trk_img(self,pram,ax,**kwargs):
        a,b,t0,x0,y0 = pram
        X,Y = np.hstack([infra.gen_ellipse(a*p.q,b*p.q,t0,x0,y0,p.phi) for p in self.points])
        if self.charge is None:
            kwargs['marker'] = '*'
        elif self.charge == 1:
            kwargs['marker'] = '^'
        elif self.charge == -1:
            kwargs['marker'] = 'v'
        else:
            kwargs['marker'] = 'o'
        ax.plot(X,Y,**kwargs)
    def classify2(self):
        ''' second attempt at the classify function''' 
        phi,q = zip(*[(p.phi,p.q) for p in self.points])
        q = np.asarray(q)
        # if the track is less than 25, don't try to classify
        if len(phi) < 25:
            self.charge =  None
            self.q = None
            self.phi = None
            return

        p_shift = 0
        if np.min(phi) < 0.1*np.pi or np.max(phi) > 2*np.pi*.9:
            p_shift = np.pi
            phi = np.mod(np.asarray(phi) + p_shift,2*np.pi)
        a = np.vstack([q**2,q,np.ones(np.size(q))]).T
        X,res,rnk,s = nl.lstsq(a,phi)
        phif = a.dot(X)
        p = 1- ss.chi2.cdf(np.sum(((phif - phi)**2)/phif),len(q)-3)

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



            
def find_rim_fringes(pt_lst,lfimg,s_width,s_num,lookahead=5,delta=10000,s=2):
    smooth_rng = s
    
    # fit the ellipse to extract from
    out = infra.fit_ellipse(pt_lst)

    #dlfimg = scipy.ndimage.morphology.grey_closing(lfimg,(1,1))
    dlfimg = lfimg
    
    # convert the parameters to parametric form
    a,b,t0,x0,y0 = infra.gen_to_parm(out.beta)


    # compute how to trim the image.  This saves computation time.
    r = int(np.max([a,b])*(1+s_width)*1.1)
    x_shift = int(x0-r)
    x_lim = int(x0+r)
    y_shift = int(y0-r)
    y_lim = int(y0+r)

    dlfimg = lfimg[y_shift:y_lim,x_shift:x_lim]

    
    # set up points to sample at
    # this will approximately  double sample.
    C = np.pi * (a+b)*(1+ (3*((a-b)/(a+b))**2)/(10+np.sqrt(4+3*((a-b)/(a+b))**2)))
    sample_count = int(np.ceil(2*C))
    theta = np.linspace(0,2*np.pi,sample_count)


    # set up all of the points to sample at in all rings.  It is
    # faster to do all the computation is one shot
    zp_all = np.hstack([(infra.gen_ellipse(*((a*ma_scale,b*ma_scale,t0,x0-x_shift,y0-y_shift,theta,))))  
                        for ma_scale in np.linspace(1-s_width,1 +s_width,s_num)])

    # extract the values at those locations from the image.  The
    # extra flipud is to take a transpose of the points to deal
    # with the fact that the definition of the first direction
    # between plotting and the image libraries is inconsistent.
    zv_all = scipy.ndimage.interpolation.map_coordinates(dlfimg,np.flipud(zp_all),order=2)

    min_vec = []
    max_vec = []
    for j,ma_scale in enumerate(np.linspace(1-s_width,1 +s_width,s_num)):
        # select out the right region
        zv = zv_all[j*sample_count:(j+1)*sample_count] 
        # smooth the curve
        zv = infra.l_smooth(zv,smooth_rng,'blackman')

        # find the peaks, the parameters here are important
        peaks = fp.peakdetect(zv,theta,lookahead,delta,True)
        # extract the maximums
        max_pk = np.vstack(peaks[0]).T
        # extract the minimums
        min_pk = np.vstack(peaks[1]).T
        
        # append to the export vectors
        min_vec.append((ma_scale,min_pk))
        max_vec.append((ma_scale,max_pk))
    
        
    return min_vec,max_vec,(a,b,t0,x0,y0)

def proc_file(fname,new_pts,search_range,bck_img=None,memory=0,s_width = .045,s_num = 110,lookahead = 5,delta = 10000,s=2):

    c_test = cine.Cine(fname)

    #compute the background
    if bck_img is None:
        bck_img = gen_bck_img(fname)




    #  h5_fname = prefix + '/' + fn + '.h5'
    #out_file = h5py.File(h5_fname,'w+')
    tm_lst = []
    trk_res_lst = []
    p_lst = []




    for lf in c_test:
        p,tm,trk_res,new_pts,tim,tam = proc_frame(new_pts,lf/bck_img,search_range,)
        p_lst.append(p)
        tm_lst.append(tm)
        trk_res_lst.append(trk_res)
        print tm, 'seconds'


def proc_frame(new_pts, img, search_range, s_width, s_num, memory=0, lookahead=5, delta=10000, s=2):
    ''' function for inner logic of loop in proc_file'''
    _t0 = time.time()


    miv,mav,p = find_rim_fringes(new_pts,img,s_width=s_width,s_num=s_num,lookahead=lookahead,delta=delta,s=s)

    tim = link_ridges(miv,search_range,memory=memory)
    tam = link_ridges(mav,search_range,memory=memory)

    tim = [t for t in tim if len(t) > 30]
    tam = [t for t in tam if len(t) > 30]

    trk_res = (zip(*[ (t.charge,t.phi) for t in tim if t.charge is not None ]),zip(*[ (t.charge,t.phi) for t in tam if t.charge is not None ]))


    _t1 = time.time()



    a,b,t0,x0,y0 = p
    # seed the next round of points
    new_pts = np.hstack([infra.gen_ellipse(*(a*t.q,b*t.q,t0,x0,y0,t.phi,)) for t in tim+tam 
                            if len(t) > 30 and 
                            t.q is not None 
                            and t.phi is not None 
                            and t.charge is not None
                            and t.charge != 0])

    return p,(_t1 - _t0),trk_res,new_pts,tim,tam


def link_ridges(vec,search_range,memory=0):
    # generate point levels from the previous steps

    levels = [[Point1D_circ(q,phi,v) for phi,v in zip(*pks)] for q,pks in vec]
    
    trks = pt.link_full(levels,2*np.pi,search_range,hash_cls = hash_line_angular,memory = memory, track_cls = lf_Track)        
    for t in trks:
        t.classify2()

    trks.sort(key=lambda x: x.phi)
    return trks
