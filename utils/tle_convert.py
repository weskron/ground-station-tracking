from sgp4.api import Satrec, jday
import numpy as np
from datetime import datetime

def tle_to_cartesian(tle_line1, tle_line2, dt=None):
    """
    Convert TLE to Cartesian position and velocity using SGP4.

    Parameters:
        tle_line1 (str): First line of TLE
        tle_line2 (str): Second line of TLE
        dt (datetime, optional): Time to propagate to (UTC).
                                 If None, uses TLE epoch.

    Returns:
        r (np.array): Position vector [x, y, z] in km
        v (np.array): Velocity vector [vx, vy, vz] in km/s
    """

    # Create satellite record
    sat = Satrec.twoline2rv(tle_line1, tle_line2)

    # If no time provided, use TLE epoch
    if dt is None:
        jd, fr = sat.jdsatepoch, sat.jdsatepochF
    else:
        jd, fr = jday(dt.year, dt.month, dt.day,
                      dt.hour, dt.minute, dt.second + dt.microsecond*1e-6)

    # Propagate
    error_code, r, v = sat.sgp4(jd, fr)

    if error_code != 0:
        raise RuntimeError(f"SGP4 error code: {error_code}")

    return np.array(r), np.array(v)