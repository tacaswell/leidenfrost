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
import cPickle

import pymongo
from pymongo import MongoClient
import bson


class BackImgClash(RuntimeError):
    pass


class ParamWrapper(object):
    '''
    Simple class to wrap configurations up nicely
    '''
    def __init__(self, config, date, processed):
        '''
        :param config: A dictionary of parameters
        :type config: `dict`
        :param date: the date of saving the parameters
        :type date: `datetime.datetime`
        :param processed: if this set of parameters has ever been run on the full data set
        :type processed: `bool`
        '''
        self.config = config
        self.date = date
        self.processed = processed


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

    def add_proc(self, cine_hash, data, **kwargs):
        '''
        Adds information about processing a data set to the data base

        :param cine_hash: A unique identifier for the data set of interest.
        :param data: a dictionary with the data in it

        `**kwargs` can be used to pass implementation specific arguements
        '''
        raise NotImplementedError('you must define this is a sub class')


class LFmongodb(LFDbWrapper):
    def __init__(self, host='10.9.8.1', port=27017, *args, **kwargs):
        LFDbWrapper.__init__(self, *args, **kwargs)
        self.connection = MongoClient(host, port)
        self.db = self.connection.LF
        self.bck_img_coll = self.db.backimg_collection
        self.bck_img_coll.ensure_index('cine', unique=True)
        pass

    def get_background_img(self, cine_hash):
        record = self.bck_img_coll.find_one({'cine': cine_hash})
        if record is None:
            return None
        return cPickle.loads(record['bck_img'])

    def store_background_img(self, cine_hash, bck_img, overwrite=False):
        # test if it exists, add that logic
        record = {'cine': cine_hash}
        record['bck_img'] = bson.binary.Binary(bck_img.dumps())
        try:
            self.bck_img_coll.insert(record)
        except pymongo.errors.DuplicateKeyError as e:
            if overwrite:
                self.rm_background_img(cine_hash)
                # recurse with out chance of hitting this again to
                self.store_background_img(cine_hash, bck_img, overwrite=False)
            else:
                raise BackImgClash(e.message)

    def rm_background_img(self, cine_hash):
        self.bck_img_coll.remove({'cine': cine_hash})

    def get_all_configs(self, cine_hash):

        raise NotImplementedError('you must define this is a sub class')

    def get_proced_configs(self, cine_hash):

        raise NotImplementedError('you must define this is a sub class')

    def get_unproced_configs(self, cine_hash):

        raise NotImplementedError('you must define this is a sub class')

    def get_config_by_key(self, key):

        raise NotImplementedError('you must define this is a sub class')

    def get_procs(self, cine_hash):

        raise NotImplementedError('you must define this is a sub class')

    def add_proc(self, cine_hash, data, **kwargs):

        raise NotImplementedError('you must define this is a sub class')
