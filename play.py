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
    print w
    values = np.convolve(w/w.sum(),s,mode='valid')[(window_len//2):-(window_len//2)]
    return values


class point:
    '''Class to encapsulate the min/max points found on a given curve '''
    def __init__(self,a,t,v):
        self.a = a                      # the scale of the curve
        self.t = t                      # the angle of the extrema 
        self.v = v                      # the value at the extrema (can probably drop this)
        self.tracks = []                # list of tracks that are trying to claim this point

    def add_track(self,track):
        '''Adds a track to the point '''
        if track in self.tracks:
            pass
        else:
            self.tracks.append(track)
    def distance(self,point):
        '''Returns the absolute value of the angular distance between
        two points mod 2\pi'''
        d = np.mod(np.abs(self.t - point.t),2*np.pi)
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
        if not point is None:
            self.add_point(point)
        self.indx = track.count
        track.count +=1
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
        ax.plot(*zip(*[(p.a,p.t) for p in self.points]),marker='x')

class hash_line:
    def __init__(self,bin_width):
        '''1D hash table for doing the ridge linking'''
        full_width = 2*np.pi
        self.boxes = [[] for j in range(0,int(np.ceil(full_width/bin_width)))]
        self.bin_width = bin_width
        self.bin_count = len(self.boxes)
        
    def add_point(self,point):
        ''' Adds a point on the hash line'''
        t = mod(point.t,2*np.pi)
        self.boxes[int(np.floor(t/self.bin_width))].append(point)
    def get_region(self,point,bbuffer = 1):
        '''Gets the region around the point'''
        box_indx = int(np.floor(self.bin_count * mod(point.t,2*np.pi)/(2*np.pi)))
        tmp_box = []
        for j in range(box_indx - bbuffer,box_indx + bbuffer + 1):
            tmp_box.extend(self.boxes[mod(j,self.bin_count)])
        return tmp_box

    
def link_points(levels,search_range = .02):
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
        print 'start level'
        cur_hash = hash_line(search_range*2)
        for p in cur_level:
            cur_hash.add_point(p)
        print 'started', len(candidate_tracks)
        while len(candidate_tracks) > 0:
            cur_track = candidate_tracks.pop()
            
            trk_pt = cur_track.last_point()
            cur_box = cur_hash.get_region(trk_pt)
            #print len(cur_box)
            if len(cur_box) ==0:
                continue

            pmin = None
            # stupidly big number
            dmin = search_range
            
            for p in cur_box:
                if p.in_track():
                    continue
                d  = trk_pt.distance(p)
                if  d < dmin:
                    dmin = d
                    pmin = p
                    
            if pmin is not None:
                cur_track.add_point(pmin)
                accepted_tracks.append(cur_track)
                
            
                
        print 'continued',len(accepted_tracks)
        for p in cur_level:
            if not p.in_track():
                new_trk = track(p)
                track_set.append(new_trk)
                accepted_tracks.append(new_trk)
                
        candidate_tracks = accepted_tracks
        

    return track_set
