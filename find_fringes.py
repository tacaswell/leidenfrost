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
import find_peaks.peakdetect as pd
import numpy.linalg as nl
import scipy.stats as ss
import time


            
def find_rim_fringes(pt_lst,lfimg,s_width,s_num,smooth_rng=2):

    
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

    ma_scale_vec = np.linspace(1-s_width,1 +s_width,s_num)
    # set up all of the points to sample at in all rings.  It is
    # faster to do all the computation is one shot
    zp_all = np.hstack([(infra.gen_ellipse(*((a*ma_scale,b*ma_scale,t0,x0-x_shift,y0-y_shift,theta,))))
                        for ma_scale in ma_scale_vec])

    # extract the values at those locations from the image.  The
    # extra flipud is to take a transpose of the points to deal
    # with the fact that the definition of the first direction
    # between plotting and the image libraries is inconsistent.
    zv_all = scipy.ndimage.interpolation.map_coordinates(dlfimg,np.flipud(zp_all),order=2)

    min_vec = []
    max_vec = []
    for j,ma_scale in enumerate(ma_scale_vec):
        # select out the right region
        zv = zv_all[j*sample_count:(j+1)*sample_count] 
        # smooth the curve
        zv = infra.l_smooth(zv,smooth_rng,'blackman')

        # find the peaks
        peaks = pd.peakdetect_parabole(zv-np.mean(zv),theta,is_ring =True)
        # extract the maximums
        max_pk = np.vstack(peaks[0]).T
        # extract the minimums
        min_pk = np.vstack(peaks[1]).T
        
        # append to the export vectors
        min_vec.append((ma_scale,min_pk))
        max_vec.append((ma_scale,max_pk))
        
        
    return min_vec,max_vec,(a,b,t0,x0,y0)

def proc_file(fname,new_pts,search_range,bck_img=None,*args,**kwargs):

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
        p,tm,trk_res,new_pts,_,_,_,_ = proc_frame(new_pts,lf/bck_img,search_range,**kwargs)
        p_lst.append(p)
        tm_lst.append(tm)
        trk_res_lst.append(trk_res)
        print tm, 'seconds'


def proc_frame(new_pts,img,s_width,s_num,search_range, **kwargs):
    ''' function for inner logic of loop in proc_file'''
    _t0 = time.time()


    miv,mav,p = find_rim_fringes(new_pts,img,s_width=s_width,s_num=s_num,**kwargs)

    tim = link_ridges(miv,search_range,**kwargs)
    tam = link_ridges(mav,search_range,**kwargs)

    tim = [t for t in tim if len(t) > 15]
    tam = [t for t in tam if len(t) > 15]

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

    return p,(_t1 - _t0),trk_res,new_pts,tim,tam,miv,mav


def link_ridges(vec,search_range,memory=0,**kwargs):
    # generate point levels from the previous steps

    levels = [[infra.Point1D_circ(q,phi,v) for phi,v in zip(*pks)] for q,pks in vec]
    
    trks = pt.link_full(levels,2*np.pi,search_range,hash_cls = infra.hash_line_angular,memory = memory, track_cls = infra.lf_Track)        
    for t in trks:
        t.classify2(**kwargs)

    trks.sort(key=lambda x: x.phi)
    return trks
