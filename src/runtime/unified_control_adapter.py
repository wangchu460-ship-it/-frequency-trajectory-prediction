"""Fail-closed adapter for UNIFIED_F_CONTROLS_V1; no legacy F fallback."""
import numpy as np
from fair_relation_controls import ARMS

def apply_control(item,arm,cache):
    if arm not in ARMS:raise ValueError(arm)
    out=dict(item)
    if arm=='NO_MESSAGE':
        out['phy_index']=np.zeros((2,0),np.int64)
        out['phy_attr']=np.zeros((0,6),np.float32)
    else:
        if cache is None:raise ValueError('Unified cache required, including POST_F')
        out['phy_index']=cache[arm+'_index'].copy()
        out['phy_attr']=cache[arm+'_attr'].copy()
    return out
