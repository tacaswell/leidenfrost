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
    w = np.exp(-((linspace(-window_len//2,window_len//2,window_len)/(window_len//4))**2)/2)
    print w
    values = np.convolve(w/w.sum(),s,mode='valid')[(window_len//2):-(window_len//2)]
    return values
