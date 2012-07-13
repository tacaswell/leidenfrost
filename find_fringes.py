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

import matplotlib.pyplot as plt

import numpy as np
import trackpy.tracking as pt
import scipy
import scipy.ndimage
import infra
import find_peaks.peakdetect as pd
import cine
import scipy.stats as ss
import time
import scipy.interpolate as si

            
def find_rim_fringes(pt_lst,lfimg,s_width,s_num,s_pix_err = 1.5,smooth_rng=2,*args,**kwargs):
    # make sure pt_lst is a well formed array
    pt_lst = np.asarray(pt_lst)

    # a really rough estimate of the circumference 
    C = np.sum(np.sqrt(np.sum(np.diff(pt_lst,axis=1)**2,axis=0)))
    # sample points at ~ 2/pix
    sample_count = int(np.ceil(C*2))

    new_pts,tck,center = infra.get_spline(pt_lst,point_count = sample_count,pix_err = s_pix_err)

    x0,y0 = center[:,0]
    new_pts -= center

    # compute theta values
    th_new = np.arctan2(*(new_pts[::-1]))
    # compute radius
    r_new = np.sqrt(np.sum(new_pts**2,axis=0)).reshape(1,-1)
    R = np.max(r_new)*(1+s_width)*1.1
    x_shift = int(x0-R)
    x_lim = int(x0+R)
    y_shift = int(y0-R)
    y_lim = int(y0+R)
    dlfimg = lfimg[y_shift:y_lim,x_shift:x_lim]

    # this will approximately  double sample.
    ma_scale_vec = np.linspace(1-s_width,1 +s_width,s_num).reshape(-1,1)

    
    r_scaled = ma_scale_vec.dot(r_new)

    X = np.cos(th_new)*r_scaled
    Y = np.sin(th_new)*r_scaled
    zp_all = np.vstack(((Y).reshape(-1),(X).reshape(-1))) + np.flipud(center) - np.array((y_shift,x_shift)).reshape(2,1)

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
        zv = infra.l_smooth(zv,smooth_rng,'blackman')

        # find the peaks
        peaks = pd.peakdetect_parabole(zv-np.mean(zv),theta,is_ring =True)
        # extract the maximums
        max_pk = np.vstack(peaks[0])
        # extract the minimums
        min_pk = np.vstack(peaks[1])
        
        # append to the export vectors
        min_vec.append((ma_scale,min_pk))
        max_vec.append((ma_scale,max_pk))
        
        
    return min_vec,max_vec,tck,center

def proc_file(fname,new_pts,bck_img=None,file_out = None,*args,**kwargs):

    c_test = cine.Cine(fname)

    #compute the background
    if bck_img is None:
        bck_img = gen_bck_img(fname)




    #  h5_fname = prefix + '/' + fn + '.h5'
    #out_file = h5py.File(h5_fname,'w+')
    tm_lst = []
    trk_res_lst = []
    p_lst = []




    for frame_num,lf in enumerate(c_test):
        print frame_num
        tm,trk_res,new_pts,tim,tam,_,_,tck,center = proc_frame(new_pts,lf/bck_img,**kwargs)

        tm_lst.append(tm)
        trk_res_lst.append(trk_res)
        print tm, 'seconds'
        if file_out is not None:
            g = file_out.create_group('frame_%05d'%frame_num)
            infra._write_frame_tracks_to_file(g,tim,tam,{})

    return tm_lst,trk_res_lst


def proc_frame(new_pts,img,s_width,s_num,search_range, **kwargs):
    ''' function for inner logic of loop in proc_file'''
    _t0 = time.time()


    miv,mav,tck,center = find_rim_fringes(new_pts,img,s_width=s_width,s_num=s_num,**kwargs)

    tim = link_ridges(miv,search_range,**kwargs)
    tam = link_ridges(mav,search_range,**kwargs)

    tim = [t for t in tim if len(t) > 15]
    tam = [t for t in tam if len(t) > 15]

    trk_res = (zip(*[ (t.charge,t.phi) for t in tim if t.charge is not None ]),zip(*[ (t.charge,t.phi) for t in tam if t.charge is not None ]))


    _t1 = time.time()



    t_q = np.array([t.q for t in tim+tam if len(t) > 30 and 
                    t.q is not None  
                    and t.phi is not None 
                    and t.charge is not None
                    and t.charge != 0])

    t_phi = np.array([t.phi for t in tim+tam if len(t) > 30 and 
                    t.q is not None  
                    and t.phi is not None 
                    and t.charge is not None
                    and t.charge != 0])
    
    # seed the next round of points
    tmp_pts = si.splev(np.mod(t_phi,2*np.pi)/(2*np.pi),tck)
    tmp_pts -= center
    th = np.arctan2(*(tmp_pts[::-1]))
    r = np.sqrt(np.sum(tmp_pts**2,axis=0))
    r *= t_q
    indx =th.argsort()
    r = r[indx]
    th = th[indx]
    new_pts = np.vstack(((np.cos(th)*r),(np.sin(th)*r))) + center

    return (_t1 - _t0),trk_res,new_pts,tim,tam,miv,mav,tck,center


def link_ridges(vec,search_range,memory=0,**kwargs):
    # generate point levels from the previous steps

    levels = [[infra.Point1D_circ(q,phi,v) for phi,v in pks] for q,pks in vec]
    
    trks = pt.link_full(levels,2*np.pi,search_range,hash_cls = infra.hash_line_angular,memory = memory, track_cls = infra.lf_Track)        
    for t in trks:
        t.classify2(**kwargs)

    trks.sort(key=lambda x: x.phi)
    return trks
