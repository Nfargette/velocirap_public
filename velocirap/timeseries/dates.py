
""" ----------------------------------------------------------------------------
Functions for time conversion
---------------------------------------------------------------------------- """

import numpy  as np
from datetime import datetime, UTC, timedelta
from pandas   import Timestamp, to_timedelta

def central_time_to_date(time, tc): 
    """ Routine that converts time in seconds since tc to datetime64 format.

    Parameters
    ----------
    time : array of float
        Time in seconds since tc.
    tc : datetime
        Central time.

    Returns
    -------
    date : array of datetime64
        Time in datetime64 format.
    """

    return tc + to_timedelta(time, unit='s')
 
    
def date_to_sec(date, tc): # Merge with load/
    """ from date to seconds since tc

    Args:
        date (np.datetime64 [ns]): array of dates
        tc (datetime): reference time

    Returns:
        array of seconds since tc
    """
    return (date - Timestamp(tc).to_datetime64()).astype(float) * 1e-9

    
def define_shared_time(var, tb, I = None): 
    """
    Find the common time window between all variables in var and return 
    sampled time array t, time step dt and datetime array date. If I is not
    provided, the length of the time array in var.B is used.

    Parameters
    ----------
    var : dict
        Dictionary with variables as values.
    tb : datetime
        Beginning of the time interval.
    I : int, optional
        Number of time samples. If None, len(var.B.t) is used.

    Returns
    -------
    t : array
        Sampled time array.
    dt : float
        Time step.
    date : array
        Datetime array.
    """
    if I is None :
        I = np.min([200_000, len(var.B.t,)])
    t0 = np.max([F.t[~np.isnan(F.y[0])][0] for F in var.values()])
    t1 = np.min([F.t[~np.isnan(F.y[0])][-1] for F in var.values()]) # Common time window                                                    
    t  = np.linspace(t0, t1, I) # Common time sample
    dt = (t1 - t0)/I
    date = central_time_to_date(t, tb)
    return t, dt, date


def np64_to_datetime(date):
    """convert np.datetime64['ns'] to datetime, returning Nan if date is 'Nan'

    Args:
        date : date in np.datetime64['ns'] format

    Returns:
        date converted to datetime object
    """

    if date == 'Nan':
        return np.nan
    else:    
        return datetime.fromtimestamp(int(date) / 1e9, UTC).replace(tzinfo=None)
np64_to_datetime = np.vectorize(np64_to_datetime)
    

def datetime_range(start, end, spacing, unit='seconds'):
    """
    Return a list of datetime objects between start and end (inclusive)
    with a constant spacing.

    Parameters
    ----------
    start : datetime
        Start of the interval.
    end : datetime
        End of the interval.
    spacing : float or int
        Step size between consecutive datetimes.
    unit : str, optional
        Unit of spacing: 'seconds', 'minutes', 'hours', or 'days'. (default: 'seconds')

    Returns
    -------
    list[datetime]
        List of datetimes between start and end (inclusive).
    """
    if start > end:
        raise ValueError("start must be before end")

    unit = unit.lower()
    if unit == 'seconds':
        delta = timedelta(seconds=spacing)
    elif unit == 'minutes':
        delta = timedelta(minutes=spacing)
    elif unit == 'hours':
        delta = timedelta(hours=spacing)
    elif unit == 'days':
        delta = timedelta(days=spacing)
    else:
        raise ValueError("unit must be one of: 'seconds', 'minutes', 'hours', 'days'")

    times = []
    current = start
    while current <= end:
        times.append(current)
        current += delta

    return times


