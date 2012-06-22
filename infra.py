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

WINDOW_DICT = {'flat':np.ones,'hanning':np.hanning,'hamming':np.hamming,'bartlett':np.bartlett,'blackman':np.blackman}


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
    
    #open the first frame and find the initial circle
    c_test = cine.Cine(fname)    
    lfimg = c_test.get_frame(0)
    if bck_img is None:
        bck_img = np.ones(lfimg.shape)
    fig = plt.figure()
    ax = fig.add_axes([.1,.1,.8,.8])
    im = ax.imshow(lfimg/bck_img)
    im.set_clim([0,2])
    ef = ellipse_fitter()
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

