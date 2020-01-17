import pytest
import sys
import os

from core.structures import HyperStorageSet, HyperParameter

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')


@pytest.fixture
def setup_store():
    test_hyp1 = HyperParameter(name='VAL_1', value=0.8, dtype='float')
    test_hyp2 = HyperParameter(name='VAL_2', value=0.3, dtype='float')
    test_hyp3 = HyperParameter(name='VAL_3', value='banana', dtype='float')
    test_hyp4 = HyperParameter(name='VAL_4', value=5, dtype='int')

    store = HyperStorageSet([test_hyp1, test_hyp2, test_hyp3, test_hyp4])

    return store


def test_load_length(setup_store):
    """
    Test to ensure that the correct number of params is being loaded.

    :param setup_store:
    :return:
    """

    hyperparam_length = len(setup_store)
    assert hyperparam_length == 4


def test_write_out(setup_store):
    """
    Test to ensure that the writing function is working as expected.

    :param setup_store:
    :return:
    """

    paths = list()

    # go through our supported formats and test writing them out
    for mode in HyperStorageSet.valid_output_types:

        gen_path = '/tests_data/store_{0}.{0}'.format(mode)
        paths.append(gen_path)
        setup_store.to_file(gen_path, mode)

    # assert that the generated files actually exist
    for path in paths:
        assert os.path.exists(path) == 1


def test_loaders(setup_store):
    """
    Test to ensure that the load process if working correctly.

    TODO: combine this with the write out test to ensure parity between output files and loaded files?

    :param setup_store:
    :return:
    """
    pass
