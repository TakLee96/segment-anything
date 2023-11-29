import numpy as np

results = np.load('vis_v5.npy', allow_pickle=True).item()
"""
>>> results.keys()
dict_keys(['100034_483681', '10005_205677', '100142_449784', '10014_1211482', '10024_490664', '100396_1228208', '100409_439507', '100428_475255', '100434_223573', '100439_194787', '100564_202221', '100678_1260457', '100798_2205315', '100909_1208726', '100958_427612', '100985_570554', '101070_472538', '101211_480925', '10123_2152925', '101354_2152170'])

>>> len(results['100034_483681'])
11

>>> results['100034_483681'][0].keys()
dict_keys(['segmentation', 'label', 'gt'])

>>> results['100034_483681'][0]['segmentation'].shape
(423, 187)

>>> results['100034_483681'][0]['label']              
0

>>> results['100034_483681'][0]['gt'].shape
(423, 187)

>>> results['100034_483681'][0]['segmentation']
array([[ True,  True,  True, ...,  True,  True,  True],
       [ True,  True,  True, ...,  True,  True,  True],
       [ True,  True,  True, ...,  True,  True,  True],
       ...,
       [ True,  True,  True, ...,  True,  True,  True],
       [ True,  True,  True, ...,  True,  True,  True],
       [ True,  True,  True, ...,  True,  True,  True]])
"""
