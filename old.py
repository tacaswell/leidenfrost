from __future__ import print_function
import cine
import hashlib
import numpy as np
import scipy.odr as sodr

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def hash_file(fname):
    """for computing hash values of files.  This is to make it easy to
   run my data base scheme with files that are on external hard drives.

   code lifted from:
   http://stackoverflow.com/a/4213255/380231
   """

    md5 = hashlib.md5()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(128 * md5.block_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def disp_frame(fname, n, bck_img=None):
    '''Displays a given frame from the file'''

    c_test = cine.Cine(fname)

    lfimg = c_test.get_frame(n)

    if bck_img is None:
        bck_img = np.ones(lfimg.shape)
    fig = plt.figure()
    ax = fig.add_axes([.1, .1, .8, .8])
    im = ax.imshow(lfimg / bck_img)
    im.set_clim([.5, 1.5])
    ax.set_title(n)
    plt.draw()


def play_movie(fname, bck_img=None):
    '''plays the movie with correction'''

    def update_img(num, F, bck_img, im, txt):
        im.set_data(F.get_frame(num) / bck_img)
        txt.set_text(str(num))
    F = cine.Cine(fname)

    if bck_img is None:
        bck_img = np.ones(F.get_frame(0).shape)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    im = ax.imshow(F.get_frame(0) / bck_img)
    fr_num = ax.text(0.05, 0.05, 0, transform=ax.transAxes)
    im.set_clim([.75, 1.25])
    prof_ani = animation.FuncAnimation(fig, update_img, len(F), fargs=(F, bck_img, im, fr_num), interval=50)
    plt.show()

    return prof_ani


def plot_plst_data(p_lst):
    ''' makes a graph for the position, radius, angle, etc from the list of ellipse parameters'''

    print('hi')
    a, b, t0, x0, y0 = zip(*p_lst)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(a, label='a')
    ax.plot(b, label='b')
    ax.set_ylabel('axis [pix]')
    ax.set_xlabel(r'frame \#')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x0, y0)
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')
    ax.set_aspect('equal')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x0, label='x0')
    ax.plot(y0, label='y0')
    ax.legend(loc=0)
    ax.set_ylabel('center location [px]')
    ax.set_xlabel('frame \#')

    x0 = np.array(x0)
    y0 = np.array(y0)
    x0_0 = x0 - np.mean(x0)
    y0_0 = y0 - np.mean(y0)
    x0_0 = x0_0 / np.sqrt(np.sum(x0_0 ** 2))
    y0_0 = y0_0 / np.sqrt(np.sum(y0_0 ** 2))

    print(sum(x0_0 * y0_0))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.step(range(len(t0)), t0, label=r'$\theta_0$')
    ax.axhline(-np.pi / 2)
    ax.axhline(np.pi / 2)
    ax.axhline(-np.pi)
    ax.axhline(np.pi)
    ax.legend(loc=0)
    ax.set_ylabel(r'$\theta_0$ [rad]')
    ax.set_xlabel('frame \#')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(10 * (np.array(t0) - np.mean(t0)), label=r'$\theta_0$')
    ax.plot(a - np.mean(a), label='a')
    ax.plot(b - np.mean(b), label='b')
    ax.set_ylabel('axis [pix]')
    ax.set_xlabel('frame \#')
    ax.legend(loc=0)
    ax.set_ylabel(r'arb')
    ax.set_xlabel('frame \#')


def do_comp():
    pass


def save_comp(fout_base, fout_path, fout_name, params):
    # check status of h5 file

    # either open or create h5 file

    pass


def plot_tracks(img, tim, tam, tck, center, min_len=0):
    fig = plt.figure()
    ax = fig.gca()
    c_img = ax.imshow(img, cmap=plt.get_cmap('jet'), interpolation='nearest')
    c_img.set_clim([.5, 1.5])
    [t.plot_trk_img(tck, center, ax, color='b', linestyle='-') for t in tam if len(t) > min_len]
    [t.plot_trk_img(tck, center, ax, color='m', linestyle='-') for t in tim if len(t) > min_len]
    plt.draw()


def set_up_efitter(fname, bck_img=None):
    ''' gets the initial path '''
    clims = [.5, 1.5]
    #open the first frame and find the initial circle
    c_test = cine.Cine(fname)
    lfimg = c_test.get_frame(0)
    if bck_img is None:
        bck_img = np.ones(lfimg.shape)
        clims = None
    fig = plt.figure()
    ax = fig.add_axes([.1, .1, .8, .8])
    im = ax.imshow(lfimg / bck_img)
    if clims is not None:
        im.set_clim(clims)
    ef = spline_fitter(ax)

    plt.draw()

    return ef


class StackStorage(object):
    """
    A class to deal with keeping track of and spitting back out the
    results of given movie.  This class also keeps track of doing mid-level
    processing and visualization.
    """
    def __init__(self, cine_fname=None, bck_img=None):
        self.back_img = bck_img
        self.cine_fname = cine_fname
        self.cine = cine.Cine(cine_fname)
        self.frames = {}
        pass

    def add_frame(self, frame_num, data):
        """ Adds a frame to the storage
        """
        self.frames[frame_num] = data
