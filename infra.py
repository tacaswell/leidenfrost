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

import scipy
import scipy.ndimage
import cine
import lf_drop.play  as lfp
import time
import h5py
import matplotlib.pyplot as plt
import numpy as np

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
    
    #open the first frame and find the initial circle
    c_test = cine.Cine(fname)    
    lfimg = c_test.get_frame(0)
    if bck_img is None:
        bck_img = np.ones(lfimg.shape)
    fig = plt.figure()
    ax = fig.add_axes([.1,.1,.8,.8])
    im = ax.imshow(lfimg/bck_img)
    im.set_clim([0,2])
    ef = lfp.ellipse_fitter()
    plt.connect('button_press_event',ef.click_event)

    
    prefix = '/media/leidenfrost_a/leidenfrost/2012-05-10'
    fn = '270C_1kh_150us_bigdrop_02.cine'
    c_test = cine.Cine(prefix + '/' + fn)
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

def proc_file(fname,new_pts,bck_img=None):
    '''Extracts the profile for every frame in the input cine '''

    #open the cine file


    c_test = cine.Cine(fname)

    #compute the background
    if bck_img is None:
        bck_img = gen_bck_img(fname)


  

    #  h5_fname = prefix + '/' + fn + '.h5'
    #out_file = h5py.File(h5_fname,'w+')
    tm_lst = []
    trk_res_lst = []
    p_lst = []
  

  
    
    for j,lf in enumerate(c_test):
        _t0 = time.time()
  

        miv,mav,p = lfp.find_rim_fringes(new_pts,lf/bck_img,.045,110,5,.15)
        a,b,t0,x0,y0 = p
        tim = lfp.link_ridges(miv,.01)
        tam = lfp.link_ridges(mav,.01)
        
        tim = [t for t in tim if len(t) > 30]
        tam = [t for t in tam if len(t) > 30]
        
        trk_res_lst.append((zip(*[ (t.charge,t.phi) for t in tim if t.charge is not None ]),zip(*[ (t.charge,t.phi) for t in tam if t.charge is not None ])))
        p_lst.append(p)
        
        _t1 = time.time()
        print (_t1 - _t0) ,"seconds"
        tm_lst.append(_t1-_t0)

        if j > 500:
            break
        # seed the next round of points
        new_pts = np.hstack([lfp.gen_ellipse(*(a*t.q,b*t.q,t0,x0,y0,t.phi,)) for t in tim+tam if len(t) > 50 and t.q is not None and t.phi is not None and t.charge !=0])

        


    return trk_res_lst,p_lst,tm_lst

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
    

