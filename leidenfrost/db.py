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

import cine.cine
from datetime import datetime
import cPickle

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
        Deletes the background image for the data set of interest.  Does nothing if
        entry does not exist.

        :param cine_hash: A unique identifier for the data set of interest.

        '''
        raise NotImplementedError('you must define this is a sub class')

    def get_all_configs(self, cine_hash):
        '''
        Returns all configurations associated with the data file of interest.

        :param cine_hash: A unique identifier for the data set of interest.
        :rtype: a list of dictionaries, one per configuration, empty if none
        '''
        raise NotImplementedError('you must define this is a sub class')

    def get_proced_configs(self, cine_hash):
        '''
        Returns all configurations associated with the data file of interest
        that have been processed at least once

        :param cine_hash: A unique identifier for the data set of interest.
        :rtype: a list of dictionaries, one per configuration, empty if none
        '''
        raise NotImplementedError('you must define this is a sub class')

    def get_unproced_configs(self, cine_hash):
        '''
        Returns all configurations associated with the data file of interest
        that have not been processed at least once

        :param cine_hash: A unique identifier for the data set of interest.
        :rtype: a list of dictionaries, one per configuration, empty if none
        '''
        raise NotImplementedError('you must define this is a sub class')

    def get_config_by_key(self, key):
        '''
        Returns all configurations associated with the data file of interest
        that have not been processed at least once

        :param key: A unique identifier for configuration of interest
        :rtype: a `dict`, `None` if key is invalid
        '''
        raise NotImplementedError('you must define this is a sub class')

    def get_procs(self, cine_hash):
        '''
        Return data on all the time the data set of interest has been processed
        '''
        raise NotImplementedError('you must define this is a sub class')

    def store_proc(self, cine_hash, config_key, file_out, **kwargs):
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

    def remove_config(self, c_id):
        '''Removes a configuration from the data base

        Parameters
        ----------
        c_id : id type
             The id of the configuration to remove
        '''
        raise NotImplementedError('you must define this is a sub class')

    def store_config(self, cine_hash, data, **kwargs):
        '''
        Adds a configuration to the store

        :param cine_hash: A unique identifier for the data set of interest.
        :param data: a dictionary with the configuration meta-data in it

        `**kwargs` can be used to pass implementation specific arguements
        '''
        raise NotImplementedError('you must define this is a sub class')

    def add_hash_lookup(self, cine_hash, path, fname):
        '''
        Adds a mapping from the `cine_hash` -> file path

        :param cine_hash: A unique identifier for the data set of interest.
        :param path: the path of the file _relative to a base path_
        :param fname: file name of data
        '''


class LFmongodb(LFDbWrapper):
    col_map = {'bck_img': 'backimg_collection',  # collection for the background images
               'movs': 'movie_collection',  # collection for pointing to movies
               'proc': 'proc_collection',  # collection for point to the results of processing a cine
               'config': 'config_collection',  # collection of configurations used for procs
               'comment': 'comment_collection',  # collection of comments on data and/or results
               'fpath': 'filepath_collection',  # collection that maps cine_hash -> file path
               }

    def __init__(self, host='10.8.0.1', port=27017, *args, **kwargs):
        LFDbWrapper.__init__(self, *args, **kwargs)
        self.connection = MongoClient(host, port)
        self.db = self.connection.LF
        self.coll_dict = {}
        self.coll_dict['bck_img'] = self.db[self.col_map['bck_img']]
        self.coll_dict['bck_img'].ensure_index('cine', unique=True)
        for f in self.col_map:
            if f is 'bck_img':
                pass
            self.coll_dict[f] = self.db[self.col_map[f]]
            self.coll_dict[f].ensure_index('cine')
        pass

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

    def get_all_configs(self, cine_hash):
        return [_ for _ in self.coll_dict['config'].find({'cine': cine_hash})]

    def get_proced_configs(self, cine_hash):
        return [_ for _ in self.coll_dict['config'].find({'$and': [{'cine': cine_hash}, {"proced": True}]})]

    def get_unproced_configs(self, cine_hash):
        return [_ for _ in self.coll_dict['config'].find({'$and': [{'cine': cine_hash}, {"proced": False}]})]

    def get_config_by_key(self, config_key):
        if isinstance(config_key, str):
            config_key = bson.objectid.ObjectId(config_key)
        return self.coll_dict['config'].find_one({'_id': config_key})

    def get_procs(self, cine_hash):

        raise NotImplementedError('you must define this is a sub class')

    def store_proc(self, cine_hash, config_key, file_out, **kwargs):
        record = {'cine': cine_hash}
        record['conig_key'] = config_key
        f_dict = file_out._asdict()
        f_dict.pop('base_path', None)  # don't want to save the base_path part
        record['out_file'] = f_dict
        record['time_stamp'] = datetime.now()
        p_id = self.coll_dict['proc'].insert(record)
        # code goes here to update the config entries
        config_record = self.coll_dict['config'].find_one({'_id': config_key})
        config_record['proc_keys'].append(p_id)
        config_record['proced'] = True
        self.coll_dict['config'].save(config_record)

    def store_config(self, cine_hash, config_dict, curves_dict, **kwargs):
        record = {'cine': cine_hash}
        record['config'] = config_dict
        record['curves'] = curves_dict
        record['time_stamp'] = datetime.now()
        record['proced'] = False
        record['proc_keys'] = []
        _id = self.coll_dict['config'].insert(record)
        return _id

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
