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
import scipy
import scipy.ndimage
reload(fp)
BPP_LOOKUP = dict({8:'uint8',16:'uint16'})

WINDOW_DICT = {'flat':np.ones,'hanning':np.hanning,'hamming':np.hamming,'bartlett':np.bartlett,'blackman':np.blackman}

def extract_image(fname):
    im = PIL.Image.open(fname)
    img_sz = im.size[::-1]

    if 277 in im.tag.keys():
        chans = im.tag[277][0]
    else:
        chans = 1
    if 258 in im.tag.keys():
        bpp = im.tag[258]
    else:
        bpp = 16
    print chans,bpp
    if chans == 1:
        return np.reshape(im.getdata(),img_sz).astype(BPP_LOOKUP(bpp[0])).T
    else:
        return np.reshape(map(lambda x: x[0],im.getdata()),img_sz).astype(BPP_LOOKUP[bpp[0]]).T

def gen_circle(x,y,r,theta =None):
    if theta is None:
        theta = np.linspace(0,2*np.pi,1000)
    return np.vstack((r*np.sin(theta) + x,r*np.cos(theta) + y))

def gen_ellipse(a,b,t,x,y,theta):
    # a is always the major axis, x is always the major axis, can be rotated away by t
    if b > a:
            tmp = b
            b = a
            a = tmp

            
    #t = np.mod(t,np.pi/2)
    r =  1/np.sqrt((np.cos(theta - t)**2 )/(a*a) +(np.sin(theta - t)**2 )/(b*b) )
    return np.vstack((r*np.cos(theta) + x,r*np.sin(theta) + y))

class ellipse_fitter:
    def __init__(self):
        self.pt_lst = []
        
        
    def click_event(self,event):
        ''' Extracts locations from the user'''
        if event.key == 'shift':
            self.pt_lst = []
            
        self.pt_lst.append((event.xdata,event.ydata))

    def get_params(self):
        return gen_to_parm(fit_ellipse(self.pt_lst))

class circ_finder:
    def __init__(self):
        self.pt_lst = []
        
    def click_event(self,event):
        ''' Extracts locations from the user'''
        if event.key == 'shift':
            self.pt_lst = []
            
        self.pt_lst.append((event.xdata,event.ydata))

    def get_params(self):
        a,b,t,x,y = gen_to_parm(fit_ellipse(np.vstack(self.pt_lst).T).beta)
        return (0,0,0,x,y)

    

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
        self.q = None
        self.phi = None

    def __len__(self):
        return len(self.points)
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
    def plot_trk(self,ax,**kwargs):
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
        t = np.mod(point.phi,2*np.pi)
        self.boxes[int(np.floor(t/self.bin_width))].append(point)
    def get_region(self,point,bbuffer = 1):
        '''Gets the region around the point'''
        box_indx = int(np.floor(self.bin_count * np.mod(point.phi,2*np.pi)/(2*np.pi)))
        tmp_box = []
        for j in range(box_indx - bbuffer,box_indx + bbuffer + 1):
            tmp_box.extend(self.boxes[np.mod(j,self.bin_count)])
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
            tmp_box.extend(self.boxes[np.mod(j,self.bin_count)])
        return tmp_box

def linear_factory(r):
    def tmp(bin_width):
        return hash_line_linear(bin_width,r)
    return tmp
    
def link_points(levels,search_range = .02,memory=0,hash_line=hash_line_angular):
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
    mem_lists = []
    for j in range(memory):
        mem_lists.append([])
        

    for cur_level in levels[1:]:
        new_mem_list = []
        accepted_tracks = []
        
        cur_hash = hash_line(search_range*2)
        for p in cur_level:
            cur_hash.add_point(p)
        accepted_tracks,new_mem_list = _find_links(candidate_tracks,cur_hash,search_range)
        if memory>0:
       
            re_mem_lists = []
            for m in mem_lists:
                tmp_len = len(m)
                tmp_accpt,tmp_mem = _find_links(m,cur_hash,search_range)
                accepted_tracks.extend(tmp_accpt)
                re_mem_lists.append(tmp_mem)



            mem_lists = re_mem_lists
            mem_lists.pop(0)
            mem_lists.append(new_mem_list)
            
        for p in cur_level:
            if not p.in_track():
                new_trk = track(p)
                track_set.append(new_trk)
                accepted_tracks.append(new_trk)
                
        candidate_tracks = accepted_tracks
        
    
    return [t for t in track_set if len(t) >0]

def _find_links(t_list,cur_hash,search_range):
    '''This is a helper function for  link points'''

    new_mem_list = []
    accepted_tracks = []

    while len(t_list) > 0:
        # select the next track
        cur_track = t_list.pop()
        # get the last point in the current track
        trk_pt = cur_track.last_point()
        # get the region of candidate points
        cur_box = cur_hash.get_region(trk_pt)
        #print len(cur_box)

        if len(cur_box) ==0:
            new_mem_list.append(cur_track)            
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
            if pmin.in_track():
                other_track = pmin.tracks[0]
                other_track.merge_track(cur_track)
                other_track.sort()
            else:
                cur_track.add_point(pmin)
                accepted_tracks.append(cur_track)
        else:
            new_mem_list.append(cur_track)            

    return accepted_tracks,new_mem_list
def find_rim_fringes(pt_lst,lfimg,s_width,s_num,lookahead = 5,delta = 10000,s=2):
    # fit the ellipse to extract from
    
    out = fit_ellipse(pt_lst)

    #dlfimg = scipy.ndimage.morphology.grey_closing(lfimg,(1,1))
    dlfimg = lfimg
    
    # convert the parameters to parametric form
    a,b,t0,x0,y0 = gen_to_parm(out.beta)

    # set up points to sample at
    C = np.pi * (a+b)*(1+ (3*((a-b)/(a+b))**2)/(10+np.sqrt(4+3*((a-b)/(a+b))**2)))


    theta = np.linspace(0,2*np.pi,np.floor(2*C).astype('int'))
    # this will approximately  double sample.
    

    min_vec = []
    max_vec = []
    for ma_scale in np.linspace(1-s_width,1 +s_width,s_num):
        # set up this steps ellipse
        p = (a*ma_scale,b*ma_scale,t0,x0,y0)
        # extract the points in the ellipse is x-y
        zp = (gen_ellipse(*(p+(theta,))))
        # extract the values at those locations from the image.  The
        # extra flipud is to take a transpose of the points to deal
        # with the fact that the definition of the first direction
        # between plotting and the image libraries is inconsistent.
        zv = scipy.ndimage.interpolation.map_coordinates(dlfimg,np.flipud(zp),order=2)
        # smooth the curve
        zv = l_smooth(zv,s,'blackman')

        # find the peaks, the parameters here are important
        peaks = fp.peakdetect(zv,theta,lookahead,delta,True)
        # extract the maximums
        max_pk = np.vstack(peaks[0]).T
        # extract the minimums
        min_pk = np.vstack(peaks[1]).T
        
        # append to the export vectors
        min_vec.append((ma_scale,min_pk))
        max_vec.append((ma_scale,max_pk))
        
    ## # debugging code
    ## figure()
    ## imshow(dlfimg)
    ## axis('equal')
    ## plot(pt_lst[0],pt_lst[1],'x')
    ## curve2 = gen_ellipse(*(gen_to_parm(out.beta)+(theta,)))
    ## plot(curve2[0],curve2[1],label='fit')
    ## lfprof = scipy.ndimage.interpolation.map_coordinates(lfimg.T,curve2)
    
        
    return min_vec,max_vec,(a,b,t0,x0,y0)

def find_fingers(x,y,rmin,rmax,s_num,lfimg,lookahead = 5,delta = 15,s = 2,theta_rng =(0,2*np.pi)):

    
    
    # set up points to sample at


    #dlfimg = scipy.ndimage.morphology.grey_closing(lfimg,(1,1))
    dlfimg = lfimg
    
    # convert the parameters to parametric form

    min_vec = []
    max_vec = []
    for r_step in np.linspace(rmin,rmax,s_num):
        theta = np.linspace(*(theta_rng + (int(ceil(2*2*r_step*np.pi)),)))
        # extract the points in the ellipse is x-y
        zp = (gen_circle(x,y,r_step,theta ) )
        # extract the values at those locations from the image.  The
        # extra flipud is to take a transpose of the points to deal
        # with the fact that the definition of the first direction
        # between plotting and the image libraries is inconsistent.
        zv = scipy.ndimage.interpolation.map_coordinates(dlfimg,flipud(zp),order=4).astype('float')
        # smooth the curve
        zv = l_smooth(zv,s)

        
        # find the peaks, the parameters here are important
        peaks = fp.peakdetect(diff(zv),theta[:-1] + mean(diff(theta))/2,lookahead,delta)
        # extract the maximums

        if len(peaks[0]) > 0:
            max_pk = np.vstack(peaks[0]).T
        else:
            max_pk = []
        # extract the minimums
        if len(peaks[1]) >0:
            min_pk = np.vstack(peaks[1]).T
        else:
            min_pk = []
        
        # append to the export vectors
        min_vec.append((r_step,min_pk))
        max_vec.append((r_step,max_pk))
                
    return min_vec,max_vec

def link_ridges(vec,search_range,memory=0):
    # generate point levels from the previous steps

    levels = [[point(q,phi,v) for phi,v in zip(*pks)] for q,pks in vec]
    
    trks = link_points(levels,search_range,memory)        
    for t in trks:
        t.classify()

    trks.sort(key = lambda x: x.phi)
    return trks


def link_fingers(vec,search_range,memory=0):
    # generate point levels from the previous steps

    levels = [[point(q,phi,v) for phi,v in zip(*pks)] for q,pks in vec]
    
    trks = link_points(levels,search_range,memory)        

    trks.sort(key = lambda x: x.phi)
    return trks


def link_rings(vec,search_range,r_max):
    # generate point levels from the previous steps
    hash_line = linear_factory(r_max)
    levels = [[point(a,t,v) for t,v in zip(*pks)] for a,pks in vec]
    
    trks = link_points(levels,search_range,hash_line = hash_line)        
    
    return trks


    
def radial_merge_tracks(trk_lst,merge_range):
    '''This function is for merging radial tracks together.  This
    assumes that the data is from concentric rings.  phi -> r, q ->
    theta.  All tracks that have an average phi with in merge_range of
    each other are merged into one track.'''
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
