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
from __future__ import division, print_function


from datetime import datetime
import cPickle
import os.path
import pymongo
from pymongo import MongoClient
import bson


class BackImgClash(RuntimeError):
    pass


class LFDbWrapper(object):
    '''
    An ABC for dealing with talking to a data base.

    For abstracting away mongo vs sqlite as I can not make up my mind.
    '''
    def __init__(self, *args, **kwargs):
        pass

    def get_background_img(self, cine_hash):
        '''
        Returns the background image for this cine if it exists, or `None` if
        one does not exist.

        :param cine_hash: A unique identifier for the data set of interest.
        :rtype: `numpy.ndarray` or `None`
        '''
        raise NotImplementedError('you must define this is a sub class')

    def store_background_img(self, cine_hash, bck_img, overwrite=False):
        '''
        Store the background image for this cine in the database.

        If an entry already exists for this file and overwrite is
        true, replace the entry, if overwrite is false, raise `BackImgClash`.

        :param cine_hash: A unique identifier for the data set of interest.
        :param bck_img: the data
        :type bck_img: `numpy.ndarray`
        :param overwrite: If existing images should be overwritten
        :type overwrite: `bool`

        '''
        raise NotImplementedError('you must define this is a sub class')

    def rm_background_img(self, cine_hash):
        '''
        Deletes the background image for the data set of interest.  Does
        nothing if entry does not exist.

        :param cine_hash: A unique identifier for the data set of
        interest.

        '''
        raise NotImplementedError('you must define this is a sub class')

    def get_procs(self, cine_hash):
        '''
        Return data on all the time the data set of interest has been processed
        '''
        raise NotImplementedError('you must define this is a sub class')

    def store_proc(self, cine_hash, parameters, file_out, **kwargs):
        '''
        Adds information about processing a data set to the data base

        :param cine_hash: A unique identifier for the data set of interest.
        :param config_key: A unique ID for the configuration used
        :param file_out: a `leidenfrost.FilePath` object

        `**kwargs` can be used to pass implementation specific arguements
        '''
        raise NotImplementedError('you must define this is a sub class')

    def remove_proc(self, proc_key):
        '''Removes a proc record from the db '''
        raise NotImplementedError('you must define this is a sub class')

    def add_hash_lookup(self, cine_hash, path, fname):
        '''
        Adds a mapping from the `cine_hash` -> file path

        :param cine_hash: A unique identifier for the data set of interest.
        :param path: the path of the file _relative to a base path_
        :param fname: file name of data
        '''
        raise NotImplementedError('you must define this is a sub class')


class LFmongodb(LFDbWrapper):
    col_map = {'bck_img': 'backimg_collection',  # collection for the
                                                 #background images
               'movs': 'movie_collection',  # collection for pointing
                                            #to movies
               'proc': 'fringe_proc_collection',  # collection for
                                                  #point to the
                                                  #results of
                                                  #processing a cine
               'RM': 'RM_proc_collection',  # collection for point to
                                            #the results of processing
                                            #a cine
               'comment': 'comment_collection',  # collection of
                                                 #comments on data
                                                 #and/or results
               }

    def __init__(self, host='10.8.0.1', port=27017, disk_dict=None,
                 *args, **kwargs):
        LFDbWrapper.__init__(self, *args, **kwargs)
        self.connection = MongoClient(host, port)
        self.db = self.connection.LF
        self.coll_dict = {}
        for f in self.col_map:
            self.coll_dict[f] = self.db[self.col_map[f]]
            if f in ['bck_img', 'movs']:
                self.coll_dict[f].ensure_index('cine', unique=True)
            else:
                self.coll_dict[f].ensure_index('cine')

        if disk_dict is None:
            # hard code in my disks
            disk_dict = {'/media/leidenfrost_a': 0,
                         '/media/tcaswell/leidenfrost_a': 0,
                         '/media/tcaswell/leidenfrost_b': 0,
                         '/media/leidenfrost_b': 0,
                         '/media/leidenfrost_c': 1,
                         '/media/tcaswell/leidenfrost_c': 1,
                         '/media/tcaswell/leidenfrost_d': 1,
                         '/media/leidenfrost_d': 1,
                         }
        self.disk_dict = disk_dict
        self.i_disk_dict = {v: k for k, v in disk_dict.items()}

        pass

    def set_disk_dict(self, disk_dict):
        """
        Sets the dictionaries used for translating disk number -> path

        """

    def store_movie_md(self, cine, cine_path,
                       calibration_value, calibration_unit):
        """
        Stores a movie in the data base

        Parameters
        ----------
        cine : `Cine` object
            an open `Cine` object
        cine_path : `FilePath`
            Where the cine is

        calibration_value : float
            The length per pixel

        calibration_unit : string
        """
        tmp_dict = {}
        # save the hash
        tmp_dict['cine'] = cine.hash
        # save the calibration
        tmp_dict['cal_val'] = calibration_value
        tmp_dict['cal_unit'] = calibration_unit
        # save the path information
        f_dict = cine_path._asdict()
        f_dict.pop('base_path', None)  # don't want to save the base_path part
        tmp_dict['fpath'] = f_dict
        # save the frame rate
        tmp_dict['frame_rate'] = cine.frame_rate
        # save the camera version
        tmp_dict['camera'] = cine.camera_version
        self.coll_dict['movs'].insert(tmp_dict)

    def get_movie_md(self, cine_hash):
        record = self.coll_dict['movs'].find_one({'cine': cine_hash})
        return record

    def get_background_img(self, cine_hash):
        record = self.coll_dict['bck_img'].find_one({'cine': cine_hash})
        if record is None:
            return None
        return cPickle.loads(record['bck_img'])

    def store_background_img(self, cine_hash, bck_img, overwrite=False):
        # test if it exists, add that logic
        record = {'cine': cine_hash}
        record['bck_img'] = bson.binary.Binary(bck_img.dumps())
        try:
            self.coll_dict['bck_img'].insert(record)
        except pymongo.errors.DuplicateKeyError as e:
            if overwrite:
                self.rm_background_img(cine_hash)
                # recurse with out chance of hitting this again to
                self.store_background_img(cine_hash, bck_img, overwrite=False)
            else:
                raise BackImgClash(e.message)

    def rm_background_img(self, cine_hash):
        self.coll_dict['bck_img'].remove({'cine': cine_hash})

    def get_procs(self, cine_hash):
        return self.coll_dict['proc'].find_one({'cine': cine_hash})

    def start_proc(self, cine_hash, parameter_dict, curve, file_out, **kwargs):
        # start with the cine hash
        record = {'cine': cine_hash}
        # time stamp
        record['start_time_stamp'] = datetime.now()
        # store parameters
        record['parameters'] = parameter_dict
        # store the seed curve
        record['curve'] = curve.to_pickle_dict
        # insert and return _id
        _id = self.coll_dict['proc'].insert(record)
        # convert the FilePath -> dict for storage
        fname, ext = os.path.splitext(file_out.fname)
        file_out = file_out._replace(fname="{}_{}{}".format(fname, _id, ext))
        f_dict = file_out._asdict()
        # map the local base_path to disk number
        f_dict['disk'] = self.disk_dict.get(f_dict.pop('base_path', None), '')
        record['out_file'] = f_dict
        self.coll_dict['proc'].save(record)
        return _id, file_out

    def finish_proc(self, id):
        record = self.coll_dict['proc'].find_one({'_id': id})
        record['done_time_stamp'] = datetime.now()
        record['finished'] = True
        self.coll_dict['proc'].save(record)

    def timeout_proc(self, id):
        record = self.coll_dict['proc'].find_one({'_id': id})
        record['timeout'] = True
        self.coll_dict['proc'].save(record)

    def flag_proc_useful(self, id):
        record = self.coll_dict['proc'].find_one({'_id': id})
        record['useful'] = True
        self.coll_dict['proc'].save(record)

    def flag_proc_useless(self, id):
        record = self.coll_dict['proc'].find_one({'_id': id})
        record['useful'] = False
        self.coll_dict['proc'].save(record)

    def remove_proc(self, proc_key):
        self.coll_dict['proc'].remove({'_id': proc_key})

    def remove_config(self, c_id):
        self.coll_dict['config'].remove({'_id': c_id})

    def add_comment(self, cine_hash, comment_dict=None, **kwargs):
        '''
        Add a comment to about a LF movie or processed result to db
        '''
        record = {'cine': cine_hash}
        record['time_stamp'] = datetime.now()
        if comment_dict is not None:
            for key, val in comment_dict.items():
                if key in record:
                    print('a')
                    pass
                record[key] = val
        for key, val in kwargs.items():
            if key in record:
                print ('duplicate key!')
            record[key] = val
        _id = self.coll_dict['comment'].insert(record)
        return _id

    def get_comment_by_id(self, _id):
        if isinstance(_id, str):
            _id = bson.objectid.ObjectId(_id)
        return self.coll_dict['comment'].find_one({'_id': _id})

    def get_comment_by_cine_hash(self, cine_hash):
        return [_ for _ in self.coll_dict['config'].find({'cine': cine_hash})]

    def set_good_frame_range(self, proc_id, start, end):
        """
        Sets the good frame range for this proc

        Parameters
        ----------
        proc_id : _uid
           The _uid of the process to attach the data to

        start : int
           The first good frame in this proc file

        end : int
           The first bad frame (or one past the end) in this file
        """
        record = self.coll_dict['proc'].find_one({'_id': proc_id})
        record['in_frame'] = start
        record['out_frame'] = end
        self.coll_dict['proc'].save(record)

    def get_proc_id(self, fname):
        """
        Given a FilePath and a cinehash, figure out the proc's id

        Parameters
        ----------
        fname : FilePath
            The path to the hdf5 file

        Returns
        -------
        id : the _id of this proc in the database

        """
        return self.coll_dict['proc'].find_one({'out_file.fname': fname.fname.strip('/'),
                                                'out_file.path': fname.path.strip('/')}
                                                )['_id']

    def get_proc_entry(self, proc_id):
        """
        Given a proc_id return the full dictionary of information stored

        Parameters
        ----------
        proc_id : ObjectID or equivalent
            The proc to pull data for

        Returns
        -------
        dict : the full set of meta-data for the proc
        """
        return self.coll_dict['proc'].find_one({'_id': proc_id})
