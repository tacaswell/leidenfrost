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
import numpy as np
import scipy.ndimage

import find_peaks as  fp


BPP_LOOKUP = dict({8:'uint8',16:'uint16'})



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

class circ_finder(object):
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

            
class hash_line_linear(object):
    '''1D hash table for doing ridge linking when sampling radially'''
    def __init__(self,max_r,bin_width):
        
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
