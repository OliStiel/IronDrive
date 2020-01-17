from core.structures import HyperStorageSet, HyperParameter, HyperLibrary

# TODO, write better examples

param1 = HyperParameter(name='Example_Param_1', value=0.8, dtype='float')
param2 = HyperParameter(name='Example_Param_2', value=5, dtype='int')
store = HyperStorageSet(params=[param1, param2])

param3 = HyperParameter(name='Example_Param_1', value=0.3, dtype='float')
param4 = HyperParameter(name='Example_Param_2', value=5, dtype='int')
store2 = HyperStorageSet(params=[param3, param4])

library = HyperLibrary([store, store2])

