#Copyright 2011 Thomas A Caswell
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

import PIL.Image
import scipy.odr as sodr
import numpy as np

import find_peaks as  fp

def extract_image(fname):
    im = PIL.Image.open(fname)
    img_sz = im.size[::-1]
    return np.reshape(im.getdata(),img_sz).astype('uint16').T

def gen_circle(x,y,r):
    theta = linspace(0,2*np.pi,1000)
    return vstack((r*sin(theta) + x,r*cos(theta) + y))

def gen_ellipse(a,b,t,x,y,theta):
    # a is always the major axis, x is always the major axis, can be rotated away by t
    if b > a:
            tmp = b
            b = a
            a = tmp

            
    #t = mod(t,np.pi/2)
    r =  1/np.sqrt((np.cos(theta - t)**2 )/(a*a) +(np.sin(theta - t)**2 )/(b*b) )
    return vstack((r*np.cos(theta) + x,r*np.sin(theta) + y))

class ellipse_fitter:
    def __init__(self):
        self.pt_lst = []
        
        
    def click_event(self,event):
        ''' Extracts locations from the user'''
        if event.key == 'shift':
            self.pt_lst = []
            
        self.pt_lst.append((event.xdata,event.ydata))

def e_funx(p,r):
    x,y = r
    a,b,c,d,f = p
        
    return a* x*x + 2*b*x*y + c * y*y + 2 *d *x + 2 * f *y -1

def fit_ellipse(r):


    p0 = (2,2,0,0,0)
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

    t0 =  (1/2) * arctan(2*b/(a-c))
    
    if a>c: 
        t0 =  (1/2) * arctan(2*b/(a-c))
        
    else:
        t0 = np.pi/2 + (1/2) * arctan(2*b/(c-a))
        
    

    return (ap,bp,t0,x0,y0)


def l_smooth(values,window_len=2):
    window_len = window_len*2+1
    s=np.r_[values[window_len-1:0:-1],values,values[-1:-window_len:-1]]
    #w = np.ones(window_len,'d')
    w = np.exp(-((linspace(-(window_len//2),window_len//2,window_len)/(window_len//4))**2)/2)
    
    values = np.convolve(w/w.sum(),s,mode='valid')[(window_len//2):-(window_len//2)]
    return values


class point:
    '''Class to encapsulate the min/max points found on a given curve 
    points are on a line parametrized as phi(q)
    '''
    def __init__(self,q,phi,v):
        self.q = q                      # paramterizing variable
        self.phi = phi                  # function_value
        self.v = v                      # the value at the extrema (can probably drop this)
        self.tracks = []                # list of tracks that are trying to claim this point

    def add_track(self,track):
        '''Adds a track to the point '''
        if track in self.tracks:
            pass
        else:
            self.tracks.append(track)
    def remove_from_track(self,track):
        '''Removes a point from the given track, error if not really
        in that track'''
        self.tracks.remove(track)
    def distance(self,point):
        '''Returns the absolute value of the angular distance between
        two points mod 2\pi'''
        d = np.mod(np.abs(self.phi - point.phi),2*np.pi)
        if d> np.pi:
            d = d-np.pi
        return d

    def in_track(self):
        '''Returns if a point is in a track '''
        if len(self.tracks)>0:
            return True
        else:
            return False
class track:
    count = 0
    def __init__(self,point=None):
        self.points = []
        # will take initiator point
        if not point is None:
            self.add_point(point)
                                        
        self.indx = track.count           #unique id
        track.count +=1
        self.charge = None
        self.q_bar = None
        self.phi = None
    def add_point(self,point):
        '''Adds a point to this track '''
        if point in self.points:
            return
        else:
            self.points.append(point)
            point.add_track(self)
    def last_point(self):
        '''Returns the last point on the track'''
        return self.points[-1]
    def plot_trk(self,ax):
        ax.plot(*zip(*[(p.q,p.phi) for p in self.points]),marker='x')
    # classify tracks
    def classify(self):
        t,a = zip(*[(p.phi,p.q) for p in self.points])
        self.phi = np.mean(t)
        if len(t) < 15:
            self.charge =  0
            return
        
        i_min = min(t)
        i_max = max(t)
        match_count = 0
        match_val = 0
        fliped = False
        while len(t) >=15:
            t = t[4:-5]
            t_min = min(t)
            t_max = max(t)
            if t_min == i_min:
                if match_val != -1:
                    if match_val == 1:
                        fliped = True
                        i_min = t_min
                    match_val = -1
                    match_count =0
                match_count +=1
            if t_max == i_max:
                if match_val != 1:
                    if match_val == -1:
                        fliped = True
                        i_max = t_max
                    match_val = 1
                    match_count =0
                match_count +=1
            if match_count == 2:
                self.charge = match_val
                if match_val == -1:
                    self.phi = i_min
                elif match_val == 1:
                    self.phi = i_max
                return
        if not fliped:
            self.charge =  match_val
            if match_val == -1:
                self.phi = i_min
            elif match_val == 1:
                self.phi = i_max
            return
        else:
            self.charge = 0
            return 
    def mean_phi(self):
        self.phi = np.mean([p.phi for p in self.points])
    def merge_track(self,to_merge_track):
        '''Merges the track add_track into the current track.
        Progressively moves points from the other track to this one.
        '''
        
        while len(to_merge_track.points) >0:
            cur_pt = to_merge_track.points.pop()
            cur_pt.remove_from_track(to_merge_track)
            self.add_point(cur_pt)
        if self.phi is not None:
            self.mean_phi()
        if self.charge is not None:
            self.classify()
    def sort(self):
        self.points.sort(key = lambda x: x.q)
class hash_line_angular:
    '''1D hash table with linked ends for doing the ridge linking
    around a rim'''
    def __init__(self,bin_width):
        
        full_width = 2*np.pi
        self.boxes = [[] for j in range(0,int(np.ceil(full_width/bin_width)))]
        self.bin_width = bin_width
        self.bin_count = len(self.boxes)
        
    def add_point(self,point):
        ''' Adds a point on the hash line'''
        t = mod(point.phi,2*np.pi)
        self.boxes[int(np.floor(t/self.bin_width))].append(point)
    def get_region(self,point,bbuffer = 1):
        '''Gets the region around the point'''
        box_indx = int(np.floor(self.bin_count * mod(point.phi,2*np.pi)/(2*np.pi)))
        tmp_box = []
        for j in range(box_indx - bbuffer,box_indx + bbuffer + 1):
            tmp_box.extend(self.boxes[mod(j,self.bin_count)])
        return tmp_box


            
class hash_line_linear:
    '''1D hash table for doing ridge linking when sampling radially'''
    def __init__(self,bin_width,max_r):
        
        full_width = max_r
        self.boxes = [[] for j in range(0,int(np.ceil(full_width/bin_width)))]
        self.bin_width = bin_width
        self.bin_count = len(self.boxes)
        
    def add_point(self,point):
        ''' Adds a point on the hash line'''
        t = point.phi
        self.boxes[int(np.floor(t/self.bin_width))].append(point)
    def get_region(self,point,bbuffer = 1):
        '''Gets the region around the point'''
        t = point.phi
        box_indx = int(np.floor(t/self.bin_width))
        min_b = box_indx - bbuffer
        max_b = box_indx + bbuffer +1
        if min_b < 0: 
            min_b = 0
        if max_b > self.bin_count:
            max_b = self.bin_count 
            
        tmp_box = []
        for j in range(min_b,max_b):
            tmp_box.extend(self.boxes[mod(j,self.bin_count)])
        return tmp_box

def linear_factory(r):
    def tmp(bin_width):
        return hash_line_linear(bin_width,r)
    return tmp
    
def link_points(levels,search_range = .02,hash_line=hash_line_angular):
    '''Stupid 1D linking routine.  The plan is to not worry about
    multiple connections at this stage and to instead write a way to
    merge tracks together.  Should be an issue with max points and saddles

    levels list of lists of point objects.  The inner list is grouped by a
    '''
    cur_level = levels[0]
    # initialize the master track list with the points in the first level
    track_set = [track(p) for p in cur_level]
    

    # 
    candidate_tracks = []
    candidate_tracks.extend(track_set)
    
    
    for cur_level in levels[1:]:
        accepted_tracks = []
        
        cur_hash = hash_line(search_range*2)
        for p in cur_level:
            cur_hash.add_point(p)
        
        while len(candidate_tracks) > 0:
            # select the next track
            cur_track = candidate_tracks.pop()
            # get the last point in the current track
            trk_pt = cur_track.last_point()
            # get the region of candidate points
            cur_box = cur_hash.get_region(trk_pt)
            #print len(cur_box)
            
            if len(cur_box) ==0:
                continue

            pmin = None
            # stupidly big number
            dmin = search_range
            
            for p in cur_box:
                # don't link previously linked particles
                if p.in_track():
                    continue
                # get distance between the current point and the candidate point
                d  = trk_pt.distance(p)
                
                if  d < dmin:
                    dmin = d
                    pmin = p
                    
            if pmin is not None:
                cur_track.add_point(pmin)
                accepted_tracks.append(cur_track)
                
            
                
                
        for p in cur_level:
            if not p.in_track():
                new_trk = track(p)
                track_set.append(new_trk)
                accepted_tracks.append(new_trk)
                
        candidate_tracks = accepted_tracks
        

    return track_set

def find_rim_fringes(pt_lst,lfimg,s_width,s_num):
    # fit the ellipse to extract from
    out = fit_ellipse(pts)
    # set up points to sample at
    theta = linspace(0,2*np.pi,floor(450*2*np.pi).astype('int'))

    #dlfimg = scipy.ndimage.morphology.grey_closing(lfimg,(1,1))
    dlfimg = lfimg
    
    # convert the parameters to parametric form
    a,b,t0,x0,y0 = gen_to_parm(out.beta)
    min_vec = []
    max_vec = []
    for ma_scale in linspace(1-s_width,1 +s_width,s_num):
        # set up this steps ellipse
        p = (a*ma_scale,b*ma_scale,t0,x0,y0)
        # extract the points in the ellipse is x-y
        zp = (gen_ellipse(*(p+(theta,))))
        # extract the values at those locations from the image.  The
        # extra flipud is to take a transpose of the points to deal
        # with the fact that the definition of the first direction
        # between plotting and the image libraries is inconsistent.
        zv = scipy.ndimage.interpolation.map_coordinates(dlfimg,flipud(zp),order=4)
        # smooth the curve
        zv = l_smooth(zv)

        # find the peaks, the parameters here are important
        peaks = fp.peakdetect(zv,theta,5,10000)
        # extract the maximums
        max_pk = np.vstack(peaks[0]).T
        # extract the minimums
        min_pk = np.vstack(peaks[1]).T
        
        # append to the export vectors
        min_vec.append((ma_scale,min_pk))
        max_vec.append((ma_scale,max_pk))

        
        
    return min_vec,max_vec

def link_ridges(vec,search_range):
    # generate point levels from the previous steps

    levels = [[point(q,phi,v) for phi,v in zip(*pks)] for q,pks in vec]
    
    trks = link_points(levels,search_range)        
    for t in trks:
        t.classify()

    trks.sort(key = lambda x: x.phi)
    return trks


def link_rings(vec,search_range,r_max):
    # generate point levels from the previous steps
    hash_line = linear_factory(r_max)
    levels = [[point(a,t,v) for t,v in zip(*pks)] for a,pks in vec]
    
    trks = link_points(levels,search_range,hash_line = hash_line)        
    
    return trks


    
def radial_merge_tracks(trk_lst,merge_range):
    hash_line = hash_line_linear(1,35)
    for t in trk_lst:
        t.mean_phi()
        hash_line.add_point(t)
    trk_lst.sort(key = lambda x: len(x.points))
    new_trk_lst = []
    while len(trk_lst)>0:
        t = trk_lst.pop(0)
        if len(t.points) > 0:
            cur_region = hash_line.get_region(t)
            for merge_cand in cur_region:
                if merge_cand != t and len(merge_cand.points )> 0  and np.abs(t.phi-merge_cand.phi)<merge_range:
                    t.merge_track(merge_cand)
                    t.sort()
                    
                    
            new_trk_lst.append(t)
    new_trk_lst.sort(key = lambda x: x.phi)
    return new_trk_lst
