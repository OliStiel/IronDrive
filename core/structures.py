"""
HyperParameter, HyperStorageSet, HyperLibrary
---------

Three separate structures for handling different levels of model configuration. An individual configuration setting is
handled via the HyperParameter object, a set of configuration settings (for example, the whole configuration for a model
run), and the HyperLibrary, which handles a collection of configuration settings (for example, a grid of configuration
settings used during training).

"""

import json
import functools
import datetime
import pathlib
import hashlib
from typing import List

import yaml
from pandas import DataFrame


class HyperParameter(object):
    """
    Defines a hyperparameter object. Stores a hyperparameter name and value, and allows for boilerplate tests to check
    if there's going to be compatibility issues down the way. Used to represent an individual hyperparameter
    """

    valid_dtypes = ['str', 'int', 'float']

    def __init__(self, name: str, value: str or int or float, dtype: str):
        self.name = name
        self.value = value
        self.type = dtype

        self._validate_dtype()

    def __str__(self):
        return "{0} | {1} | {2}".format(self.name, self.value, self.type)

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)

    def __repr__(self):

        if self.type == 'str':
            repr_val = "'{0}'".format(self.value)
        else:
            repr_val = self.value

        attr_str = "name='{0}', value={1}, dtype={2}".format(self.name, repr_val, self.type)

        return '{0}.{1}({2})'.format(self.__class__.__module__,
                                     self.__class__.__qualname__,
                                     attr_str)

    def __len__(self):
        return len(self.value)

    def _validate_dtype(self):
        """
        Check to ensure that the provided data type falls within expected data types.

        :return:
        """

        if self.type not in self.valid_dtypes:
            raise ValueError('Provided data type ({0}) is not an accepted Python dtype'.format(self.type))

        elif type(self.value).__name__ != self.type:
            raise ValueError('Value ({0}) type does not match provided data type! ({1})'.format(self.value, self.type))

    def return_as_dict(self) -> dict:
        """
        Returns the hyperparameter as a dictionary.

        :return:
        """
        
        return {self.name: {
            'value': self.value,
            'dtype': self.type
        }}


class HyperStorageSet(object):
    """
    Defines a collection of hyperparameters with their associated types. Used to define a set of hyperparameters that
    might be used for a single model run.
    """

    valid_output_types = ['json', 'yaml']

    def __init__(self, params: str or dict or list, set_name=None):
        """

        :param params: str, dict or list
        :param set_name: str, unique name of the set to be used for fingerprinting
        """

        self.ID = set_name
        self.score = None

        if isinstance(params, list):
            self._param_array = params

        if isinstance(params, str):

            ext = pathlib.Path(params).suffix

            funcmap = {
                '.yaml': self._from_yaml,
                '.json': self._from_json}

            self._param_array = funcmap[ext](params)

        if isinstance(params, dict):
            self._param_array = self._from_dict(params)

        # shunt those dictionary keys out to attributes
        for params in self._param_array:
            setattr(self, params.name, params.value)

        if not self.ID:
            self.ID = self.hash_contents()

    def __str__(self):
        return '\nSet Name: {0} \n ===========================\n\t'.format(self.ID) + \
               '\n\t'.join([str(hyperparam) for hyperparam in self._param_array])

    def __len__(self):
        return len(self._param_array)

    def __iter__(self):
        return iter(self._param_array)

    def __eq__(self, other):

        try:
            if self.hash_contents() == other.hash_contents():
                return True

            else:
                return False
        except AttributeError:
            raise AttributeError("Attempted to compare HyperStorageSet to "
                                 "object of type '{0}', which is not valid".format(type(other)))

    def __contains__(self, item):
        if item in self._param_array:
            return True

        else:
            return False

    @property
    def names(self):
        return [x.name for x in self._param_array]

    @property
    def values(self):
        return [x.value for x in self._param_array]

    def hash_contents(self) -> str:
        """
        Return the hashed contents of the parameter array for the purpose of comparisons and fingerprinting later on
        in the process.

        :return: str, hashed contents of the array

        >>> param1 = HyperParameter(name='Example_Param_1', value=0.8, dtype='float')
        >>> param2 = HyperParameter(name='Example_Param_2', value=5, dtype='int')
        >>> store = HyperStorageSet(params=[param1, param2])
        >>> print(store.hash_contents())
        1c97e6c8adc8f8be8a800efdacf394db0f39f2ab
        """

        shash = hashlib.sha1()
        shash.update(str(self._param_array).encode('utf-8'))

        return shash.hexdigest()

    def get_config_names(self) -> list:
        """
        Return config names as a list

        :return: list
        """

        return [x.name for x in self._param_array]

    def to_dict(self) -> dict:
        """
        Return the contents of the store as a dictionary.

        :return: dict, stored hyperparameters

        >>> param1 = HyperParameter(name='Example_Param_1', value=0.8, dtype='float')
        >>> param2 = HyperParameter(name='Example_Param_2', value=5, dtype='int')
        >>> HyperStorageSet(params=[param1, param2]).to_dict()
        {'Example_Param_1': {'value': 0.8, 'dtype': 'float'}, 'Example_Param_2': {'value': 5, 'dtype': 'int'}}


        """

        return functools.reduce(lambda x, y: {**x, **y}, [hparam.return_as_dict() for hparam in self._param_array])

    def to_file(self, fp: str, mode='json'):
        """
        Shunt stored parameters out to a file of specified type.

        :param fp: str, filepath
        :param mode: str, either json or yaml
        :return: None
        """

        funcmap = {
            'json': self._to_json,
            'yaml': self._to_yaml
        }

        funcmap[mode](fp)

        return None

    def _to_json(self, fp: str):
        """
        Generates a JSON file from the stored hyperparameters.

        :return: None
        """

        # dump them out to JSON
        with open(fp, mode='w') as file_handle:
            json.dump(obj=self.to_dict(), fp=file_handle, indent=4)

        return None

    def _to_yaml(self, fp: str):
        """
        Generates a YAML file from the stored hyperparameters

        :param fp:
        :return: None
        """

        with open(fp, mode='w+') as file_buffer:
            yaml.dump(data=self.to_dict(), stream=file_buffer)

        return None

    @staticmethod
    def _from_dict(diction: dict):
        """
        Function to generate a HyperStorage object from a dictionary

        :param diction: dict, dictionary of values
        :return: list, of HyperParameters
        """

        hyper_array = list()
        for key, value in diction.items():

            if value['dtype'] not in HyperParameter.valid_dtypes:
                raise ValueError('Hyperparameter file contains unrecognised data type: {0}'.format(value['dtype']))

            elif value['dtype'] == 'int' and '.' in value['value']:
                raise ValueError('Data type of int passed, but hyperparam is likely a float: {0}'
                                 .format(value['value']))

            else:
                conv_type = eval(value['dtype'])
                hyper_array.append(HyperParameter(name=key, value=conv_type(value['value']), dtype=value['dtype']))

        return hyper_array

    def _from_json(self, fp: str):
        """
        Generates a HyperStorage object with associated HyperParameters from a JSON file.

        :param fp: location of the json file
        :return: list, array of our hyperparameter objects
        """

        with open(fp, 'r') as file_handle:
            j_loaded = json.load(fp=file_handle, parse_constant=str, parse_int=str, parse_float=str)

        return self._from_dict(j_loaded)

    def _from_yaml(self, fp: str, loader=yaml.BaseLoader):
        """
        Generates a HyperStorageSet from a YAML file.

        :param fp: str
        :param loader: yaml.Loader
        :return: list, array of HyperParameter objects representing values
        """

        with open(fp, mode='r') as file_buffer:
            y_loaded = yaml.load(stream=file_buffer, Loader=loader)

        return self._from_dict(y_loaded)

    def log_params(self, file_path: str):
        """
        Write out a log containing the hyperparameters and the timestamp.

        :return:
        """

        now = datetime.datetime.now()

        with open(file=file_path, mode='a') as file_buffer:
            file_buffer.write('{0}| {1}\n'.format(now, ', '.join([str(x) for x in self._param_array])))


class HyperLibrary(object):
    """
    Object to handle storage of HyperStorageSets. Broadly speaking, assigns storage sets a unique fingerprint based
    on their hash, and stores them internally.

    """

    def __init__(self, sets: List[HyperStorageSet]):
        self.library = dict()

        for hset in sets:
            if hset.ID in self.library.keys():
                raise KeyError('Duplicate configuration set found for ID: {0}'.format(hset.ID))
            else:
                self.library[hset.ID] = hset

    def _framable_library(self) -> dict:
        """
        Generate a dictionary that turns into a dataframe a bit better

        :return: dict
        """

        f_dict = dict(zip(self.library.keys(), [x.values for x in self.library.values()]))
        # f_dict = {key: value.values for key, value in self.library.items()}

        return f_dict

    def get_by_fingerprint(self, fingerprint: str):
        return self.library[fingerprint]

    def to_dataframe(self):
        """
        Write out all configuration settings as a dataframe.

        :return: DataFrame
        """

        df = DataFrame.from_dict(
            data=self._framable_library(),
            orient='index',
            columns=list(self.library.values())[0].get_config_names())

        df.index = df.index.rename('ID')

        return df

    def _pprint_scores_and_config(self):
        pass


if __name__ == '__main__':
    import doctest
    doctest.testmod()
