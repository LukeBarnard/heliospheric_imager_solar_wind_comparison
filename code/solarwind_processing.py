# Data manipulation
import numpy as np
import scipy.stats as st
from scipy import interpolate
import pandas as pd
# Space science specific
import astropy.units as u
from astropy.time import Time, TimeDelta
from stereo_spice.coordinates import StereoSpice
import sunpy.sun as sun
# Color schemes
from palettable.colorbrewer.qualitative import Set1_6
from palettable.colorbrewer.sequential import YlGnBu_6
# File handling
import h5py
import os
import glob

spice = StereoSpice()

def project_info():
    """
    Return a dictionary of project directories, pointing to the different data files and folders, stored
    in config.txt
    :return proj_dirs: Dictionary of project directories, with keys 'data','figures','code', and 'results'.
    """
    files = glob.glob('config.txt')

    if len(files) != 1:
        # If too few or too many config files, guess projdirs
        print('Error: Cannot find correct config file with project directories. Check config.txt exists')
    else:
        # Open file and extract
        with open(files[0], "r") as f:
            lines = f.read().splitlines()
            proj_dirs = {l.split(',')[0]: l.split(',')[1] for l in lines}
            # Update the paths with the root directory (except for root!).
            for k in proj_dirs.iterkeys():
                if k != "root":
                    proj_dirs[k] = os.path.join(proj_dirs['root'], proj_dirs[k])

    # Just check the directories exist.
    for val in proj_dirs.itervalues():
        if not os.path.exists(val):
            print('Error, invalid path, check config: '+val)

    return proj_dirs


def get_craft_colors():
    """
    Return 3 colors used for STEREO-A, STEREO-B, and WIND in each plot.
    :return:
    """
    craft_cols = [Set1_6.mpl_colors[i] for i in [4, 3, 0]]
    return craft_cols


def get_pa_colors():
    """
    Return colors used to highlight different position angles in plots.
    :return:
    """
    pa_cols = [Set1_6.mpl_colors[i] for i in [1, 3, 4]]
    return pa_cols


def get_lagged_corr_colors():
    """
    Return colors used to produce lagged correlation plot
    :return:
    """
    return YlGnBu_6


def wind_orbit_data():
    """
    Load in the WIND ephemeris data (in GSE coords), compute equivalent HEEQ and CARR coords, and save to CSV for
    easier/quicker loading.
    """
    # Get labels for referencing the coords.
    gse_labels = ["GSE {}".format(l) for l in ['x', 'y', 'z']]
    heeq_labels = ["HEEQ {}".format(l) for l in ['x', 'y', 'z']]
    carr_labels = ["CARR {}".format(l) for l in ['x', 'y', 'z']]

    # Load in the orbit data for wind
    proj_dirs = project_info()
    orbit = pd.read_csv(proj_dirs["wind_ephemeris_data"], delimiter="    ",
                        header=None, names=['time'] + gse_labels, skiprows=31)
    # Convert date strings to datetimes
    orbit['time'] = pd.to_datetime(orbit['time'], format="%y/%m/%d %H:%M:%S")
    # Convert from Re to km
    orbit.loc[:, gse_labels] = orbit.loc[:, gse_labels] * u.earthRad.to('km')

    # Add in HEEQ and CARR coords.
    time = Time(orbit['time'].dt.to_pydatetime())
    gse_rec = orbit.loc[:, gse_labels].as_matrix()
    # Convert to HEEQ and stash
    dst_rec = spice.convert_coord(time, gse_rec, "GSE", "HEEQ")
    for i, label in enumerate(heeq_labels):
        orbit[label] = dst_rec[:, i]

    # Convert to CARR and stash
    dst_rec = spice.convert_coord(time, gse_rec, "GSE", "CARR")
    for i, label in enumerate(carr_labels):
        orbit[label] = dst_rec[:, i]

    out_path = os.path.join(proj_dirs['data'], "WIND_ORBIT_DATA.csv")
    orbit.to_csv(out_path)


def get_wind_coords(time, system):
    """
    Return an array of cartesian coordinates for WIND, interpolated at times given in time, for a coordinate system
    specified by system. time should be an astropy.time.Time object, and system is either of the strings GSE, HEEQ
    or CARR, specifiying Geocentric Solar Ecliptic, Heliospheric Earth Equatorial or Carrington coordinates.
    """
    # Load in ephmeris data
    proj_dirs = project_info()
    orbit_path = os.path.join(proj_dirs['data'], "WIND_ORBIT_DATA.csv")
    orbit = pd.read_csv(orbit_path, parse_dates=['time'])
    # Get the coords of selected system and times
    time_src = Time(orbit['time'].dt.to_pydatetime())

    if system in ["GSE", "HEEQ", "CARR"]:
        coord_labels = ["{} {}".format(system, i) for i in ['x', 'y', 'z']]
        coords = orbit.loc[:, coord_labels].as_matrix()
    else:
        print "Invalid system parsed. Should be GSE, HEEQ, or CARR. Returning NaNs"
        coords = np.zeros((time_src.size, 3)) * np.NaN

    # Interpolate the coords onto the provided times.
    coords_out = np.zeros((time.size, 3))
    # Interpolate on julian dates, as datetimes are tricky.
    # Do the interpolation on julian dates, as interpolating datetimes is ... not good.
    for i in range(coords.shape[1]):
        f = interpolate.interp1d(time_src.jd, coords[:, i], bounds_error=False)
        coords_out[:, i] = f(time.jd)

    # Remove any singleton dimension (calls for a single time point)
    coords_out = coords_out.squeeze()
    return coords_out


def get_wind_lonlat(time, system):
    """
    Returns the radius (km), longitude and latitude (degrees) of WIND for the coordinate system specified by system
    at times given in time. time should be an astropy.time.Time object, and system either of the strings GSE, HEEQ, or
    CARR
    """
    coords = get_wind_coords(time, system)
    # convert to lonlat
    lonlat = spice.convert_rec2lat(coords)
    if system == "CARR":
        # Correct the longitude so it runs from 0-360 for Carrington coords.
        id_lo = lonlat[:, 1] < 0
        if np.any(id_lo):
            lonlat[id_lo, 1] = 360.0 + lonlat[id_lo, 1]

    return lonlat


def import_wind_data():
    """
    Function to load in the WIND Solar Wind plasma data. This is hourly resolution data downloaded from SPDF? Further
    to loading in the raw data, it also adds some extra variables regarding the monitors position and timing of the
    observations in different frames of reference.
    :return:
    """
    # First process the WIND orbit data, as cannot find a spice kernel for WIND.
    wind_orbit_data()

    # Get paths to data
    proj_dirs = project_info()
    # Import wind data
    col_name = ['yr', 'doy', 'hr', 'V', 'Vaz', 'Vel', 'rho', 'T', 'sigV', 'sigVaz', 'sigVel', 'sigrho', 'sigT']
    data = pd.read_csv(proj_dirs["wind_sw_data"], delim_whitespace=True, names=col_name,
                       na_values=[9999, 999.9, 9999999])

    # Loose rows with invalid speeds
    data.dropna(axis=0, subset=['V'], inplace=True)

    # Convert yr, doy, hr to datetime.
    datestrings = []
    for i, row in data.iterrows():
        datestrings.append("{:04d}-{:03d}-{:02d}".format(int(row['yr']), int(row['doy']), int(row['hr'])))

    data['time'] = pd.to_datetime(datestrings, format="%Y-%j-%H")
    data['jd'] = Time(data['time'].dt.to_pydatetime()).jd
    data['cr'] = get_carr_num(Time(data['time'].dt.to_pydatetime()), "ert")
    data.drop(['yr', 'doy', 'hr'], axis=1, inplace=True)

    # Add in the Carrington coords for the observation
    t = Time(data['time'].dt.to_pydatetime())

    carr_lonlat = get_wind_lonlat(t, "CARR")
    # Update dataframe
    data['cr_rad'] = carr_lonlat[:, 0]
    data['cr_lon'] = carr_lonlat[:, 1]
    data['cr_lat'] = carr_lonlat[:, 2]

    # Estimate Carrignton coords of source region at 20Rs - simple backwards propagation
    data['dr'] = data['cr_rad'] * u.km - (20.0 * sun.constants.radius.to('km'))
    dt = np.int64((data['dr'] * u.km) / (data['V'].values * (u.km / u.s)).value)
    data['dt'] = pd.to_timedelta(dt, 's')
    data['t_src'] = data['time'] - data['dt']
    t_src = Time(data['t_src'].dt.to_pydatetime())
    data['jd_src'] = t_src.jd

    # Get to get Carrington lon of source
    # use HCI coords of observation with estimated source time to convert to carrington.
    obs_hae = spice.get_lonlat(t, 'earth', system='hae')
    # Convert these into Carrington coords at source time
    obs_src_car = spice.convert_lonlat(t_src, obs_hae, 'hae', 'carr')

    data['cr_lon_src'] = obs_src_car[:, 1]

    # Compute carrington time of source, make sure to correct the spill over when diff becomes negative
    dlon = data['cr_lon_src'] - data['cr_lon']
    id_lo = dlon < 0
    data['cr_src'] = data['cr'] - (dlon / 360.0)
    data.loc[id_lo, 'cr_src'] = data.loc[id_lo, 'cr_src'] - 1.0

    # Compute the appropriate position angle to look at in the STEREO A images.
    data = get_sta_carrington_pa_coords(data)
    return data


def import_sta_data():
    """
    Function to load in the STA Solar Wind plasma data. This is hourly resolution data downloaded from SPDF? In addition
    to loading in the raw data, it also adds some extra variables regarding the monitors position and timing of the
    observations in different frames of reference.
    :return:
    """

    # Get paths to data
    proj_dirs = project_info()
    # Import STA data
    col_name = ['yr', 'doy', 'hr', 'r', 'hgilat', 'hgilon', 'V', 'Vaz', 'Vel', 'rho', 'T']
    data = pd.read_csv(proj_dirs["sta_sw_data"], delim_whitespace=True, names=col_name,
                       na_values=[9999.9, 9999.9, 9999.9, 999.99, 9999999.])

    # Loose rows with invalid speeds
    data.dropna(axis=0, subset=['V'], inplace=True)

    # Convert yr, doy, hr to datetime.
    datestrings = []
    for i, row in data.iterrows():
        datestrings.append("{:04d}-{:03d}-{:02d}".format(int(row['yr']), int(row['doy']), int(row['hr'])))

    data['time'] = pd.to_datetime(datestrings, format="%Y-%j-%H")
    data['jd'] = Time(data['time'].dt.to_pydatetime()).jd
    data['cr'] = get_carr_num(Time(data['time'].dt.to_pydatetime()), "sta")
    data.drop(['yr', 'doy', 'hr'], axis=1, inplace=True)

    # Add in the Carrington coords for the observation
    t = Time(data['time'].dt.to_pydatetime())
    obs_pos = spice.get_lonlat(t, 'sta', system='carr')
    data['cr_rad'] = obs_pos[:, 0]
    data['cr_lon'] = obs_pos[:, 1]
    data['cr_lat'] = obs_pos[:, 2]

    # Estimate Carrignton coords of source region at 20Rs - simple backwards propagation
    data['dr'] = data['cr_rad'] * u.km - (20.0 * sun.constants.radius.to('km'))
    dt = np.int64((data['dr'] * u.km) / (data['V'].values * (u.km / u.s)).value)
    data['dt'] = pd.to_timedelta(dt, 's')
    data['t_src'] = data['time'] - data['dt']
    t_src = Time(data['t_src'].dt.to_pydatetime())
    data['jd_src'] = t_src.jd

    # Get to get Carrington lon of source,
    # use HCI coords of observation with estimated source time to convert to carrington.
    obs_hae = spice.get_lonlat(t, 'sta', system='hae')
    # Convert these into Carrington coords at source time
    obs_src_car = spice.convert_lonlat(t_src, obs_hae, 'hae', 'carr')

    data['cr_lon_src'] = obs_src_car[:, 1]

    # Compute carrington time of source, make sure to correct the spill over when diff becomes negative
    dlon = data['cr_lon_src'] - data['cr_lon']
    id_lo = dlon < 0
    data['cr_src'] = data['cr'] - (dlon / 360.0)
    data.loc[id_lo, 'cr_src'] = data.loc[id_lo, 'cr_src'] - 1.0

    # Compute the appropriate position angle to look at in the STEREO A images.
    data = get_sta_carrington_pa_coords(data)
    return data


def import_stb_data():
    """
    Function to load in the STB Solar Wind plasma data. This is hourly resolution data downloaded from SPDF? In addition
    to loading in the raw data, it also adds some extra variables regarding the monitors position and timing of the
    observations in different frames of reference.
    :return:
    """

    # Get paths to data
    proj_dirs = project_info()
    # Import STB data
    col_name = ['yr', 'doy', 'hr', 'r', 'hgilat', 'hgilon', 'V', 'Vaz', 'Vel', 'rho', 'T']
    data = pd.read_csv(proj_dirs["stb_sw_data"], delim_whitespace=True, names=col_name,
                       na_values=[9999.9, 9999.9, 9999.9, 999.99, 9999999.])

    # Loose rows with invalid speeds
    data.dropna(axis=0, subset=['V'], inplace=True)

    # Convert yr, doy, hr to datetime.
    datestrings = []
    for i, row in data.iterrows():
        datestrings.append("{:04d}-{:03d}-{:02d}".format(int(row['yr']), int(row['doy']), int(row['hr'])))

    data['time'] = pd.to_datetime(datestrings, format="%Y-%j-%H")
    data['jd'] = Time(data['time'].dt.to_pydatetime()).jd
    data['cr'] = get_carr_num(Time(data['time'].dt.to_pydatetime()), "stb")
    data.drop(['yr', 'doy', 'hr'], axis=1, inplace=True)

    # Add in the Carrington coords for the observation
    t = Time(data['time'].dt.to_pydatetime())
    obs_pos = spice.get_lonlat(t, 'stb', system='carr')
    data['cr_rad'] = obs_pos[:, 0]
    data['cr_lon'] = obs_pos[:, 1]
    data['cr_lat'] = obs_pos[:, 2]

    # Estimate Carrignton coords of source region at 20Rs - simple backwards propagation
    data['dr'] = data['cr_rad'] * u.km - (20.0 * sun.constants.radius.to('km'))
    dt = np.int64((data['dr'] * u.km) / (data['V'].values * (u.km / u.s)).value)
    data['dt'] = pd.to_timedelta(dt, 's')
    data['t_src'] = data['time'] - data['dt']
    t_src = Time(data['t_src'].dt.to_pydatetime())
    data['jd_src'] = t_src.jd

    # Get to get Carrington lon of source,
    # use HCI coords of observation with estimated source time to convert to carrington.
    obs_hae = spice.get_lonlat(t, 'stb', system='hae')
    # Convert these into Carrington coords at source time
    obs_src_car = spice.convert_lonlat(t_src, obs_hae, 'hae', 'carr')

    data['cr_lon_src'] = obs_src_car[:, 1]

    # Compute carrington time of source, make sure to correct the spill over when diff becomes negative
    dlon = data['cr_lon_src'] - data['cr_lon']
    id_lo = dlon < 0
    data['cr_src'] = data['cr'] - (dlon / 360.0)
    data.loc[id_lo, 'cr_src'] = data.loc[id_lo, 'cr_src'] - 1.0

    # Compute the appropriate position angle to look at in the STEREO A images.
    data = get_sta_carrington_pa_coords(data)
    return data


def load_wind_data():
    """
    Function to load in the mapped ACE Solar Wind plasma data.
    :return:
    """
    # Get paths to data
    proj_dirs = project_info()
    # Import ace data
    src_name = "wind_mapped_speed_data_20080101_20121231.csv"
    src_path = os.path.join(proj_dirs['data'], src_name)
    data = pd.read_csv(src_path, converters={"time": pd.to_datetime, "t_src": pd.to_datetime, "dt": pd.to_timedelta,
                                             "sta_time": pd.to_datetime})
    data['T'] = data['T'] / 1e6
    return data


def load_sta_data():
    """
    Function to load in the mapped ACE Solar Wind plasma data.
    :return:
    """
    # Get paths to data
    proj_dirs = project_info()
    # Import ace data
    src_name = "sta_mapped_speed_data_20080101_20121231.csv"
    src_path = os.path.join(proj_dirs['data'], src_name)
    data = pd.read_csv(src_path, converters={"time": pd.to_datetime, "t_src": pd.to_datetime, "dt": pd.to_timedelta,
                                             "sta_time": pd.to_datetime})
    data['T'] = data['T'] / 1e6
    return data


def load_stb_data():
    """
    Function to load in the mapped ACE Solar Wind plasma data.
    :return:
    """
    # Get paths to data
    proj_dirs = project_info()
    # Import ace data
    src_name = "stb_mapped_speed_data_20080101_20121231.csv"
    src_path = os.path.join(proj_dirs['data'], src_name)
    data = pd.read_csv(src_path, converters={"time": pd.to_datetime, "t_src": pd.to_datetime, "dt": pd.to_timedelta,
                                             "sta_time": pd.to_datetime})
    data['T'] = data['T'] / 1e6
    return data


def get_carr_num(time, body):
    """
    Function to compute the carrington rotation number of a solar system body or
    observatory. Hacked from SolarSoft.
    """
    carr = sun.carrington_rotation_number(time)
    he_lon, he_lat = sun.heliographic_solar_center(time)

    int_carr = np.fix(carr)
    frac_carr = carr - int_carr
    frac_lon = (360.0 - he_lon.degree) / 360.0
    cross_for = (np.abs(frac_carr - frac_lon) > (12.0 / 360.0)) & (frac_carr > frac_lon)
    cross_rev = (np.abs(frac_carr - frac_lon) > (12.0 / 360.0)) & (frac_carr < frac_lon)
    if np.any(cross_for):
        int_carr[cross_for] += 1
    if np.any(cross_rev):
        int_carr[cross_rev] -= 1

    carr_approx = int_carr + frac_lon

    ert = spice.get_lonlat(time, "ERT", "CARR")
    if ert.ndim == 1:
        ert_lon = ert[1]
    else:
        ert_lon = ert[:, 1]

    id_low = ert_lon < 0
    if np.any(id_low):
        ert_lon[id_low] = ert_lon[id_low] + 360.0

    frac = 1.0 - (ert_lon / 360.0)

    n_carr = np.round(carr_approx - frac)
    diff = carr_approx - frac - n_carr
    max_diff = np.max(np.abs(diff))
    if body in ['earth', 'ert', 'ERT']:
        carr_rot = n_carr + frac
    else:
        carr_rot = n_carr + frac
        body_coords = spice.get_lonlat(time, body, "HEEQ")
        if body_coords.ndim == 1:
            body_lon = body_coords[1]
        else:
            body_lon = body_coords[:, 1]

        carr_rot = carr_rot - (body_lon / 360.0)

    return carr_rot


def get_sta_carrington_pa_coords(data):
    """
    Fucntion to compute the position angle coordinates of another spacecraft monitor in the STA HI FOV.
    This works by computing the carrington longitude of the plane of sky low in the heliosphere. This is used to match
    the source emission time of the in-situ obs and the remote sensing obs. Then this fixed spatial point is converted
    into a position angle in the HI FOV.
    :param data:
    :return:
    """

    times = data['time']
    sta = pd.DataFrame({'time': times})
    times = Time(sta['time'].dt.to_pydatetime())
    sta['cr_num'] = get_carr_num(times, 'sta')
    sta['pos_cr_num'] = sta['cr_num'] + 0.25
    sta['jd'] = Time(sta['time'].dt.to_pydatetime()).jd

    # Do the interpolation on julian dates, as interpolating datetimes is ... not good.
    x = sta['pos_cr_num'].values
    y = sta['jd'].values
    f = interpolate.interp1d(x, y, bounds_error=False)

    for i, row in data.iterrows():
        try:
            new_jd = f(row['cr_src'])
            data.loc[i, 'sta_jd'] = new_jd
            data.loc[i, 'sta_time'] = Time(new_jd, format='jd').datetime
            src_car = [20.0 * sun.constants.radius.to('km').value, row['cr_lon_src'], row['cr_lat']]
            src_hpc = spice.convert_lonlat(Time(new_jd, format='jd'), src_car, 'carr', 'hpc', observe_dst='sta')
            el, pa = spice.convert_hpc_to_hpr(src_hpc[1], src_hpc[2])
            data.loc[i, 'sta_pa'] = pa
        except:
            data.loc[i, 'sta_jd'] = np.NaN
            data.loc[i, 'sta_time'] = pd.NaT
            data.loc[i, 'sta_pa'] = np.NaN


    return data


def situ_daily_averaging(data, reference_time="2008-01-01T00:00:00"):
    """
    Compute daily mean dataframe from an higher frequency data frame of in-situ solar wind observations. Designed to
    work with the dataframes produced by my own load functions, and does not use pandas's native *groupby* methods, as
    these don't easily cope with missing or invalid data. Instead averages the data in days since the begining of the
    record, skipping over NaNs.
    :param data: pandas dataframe of an observatories time series. Must have columns days, V, T and rho, otherwise will
                fail to compute the average quantities.
    :param reference_time: A time string from which the days shall be counted for the daily means.
    :return:
    """

    # TODO: Add checks on the dataframe that it has the right keys to calculate the daily mean properties
    t_start = Time(reference_time)
    data['days'] = data['jd'] - t_start.jd

    day_list = np.arange(data['days'].min(), data['days'].max() + 1.0, 1.0)
    dt = 0.5

    vel = np.zeros(day_list.shape)*np.NaN
    rho = np.zeros(day_list.shape)*np.NaN
    temp = np.zeros(day_list.shape)*np.NaN
    pa = np.zeros(day_list.shape) * np.NaN

    for i, day in enumerate(day_list):
        id_day = (data['days'] > (day - dt)) & (data['days'] <= (day + dt))

        # If some entries found for this day, compute the mean
        if np.any(id_day):

            # Check not all V, rho and T data are bad
            if not np.all(data.loc[id_day, 'V'].isnull()):
                vel[i] = data.loc[id_day, 'V'].mean(skipna=True)

            if not np.all(data.loc[id_day, 'rho'].isnull()):
                rho[i] = data.loc[id_day, 'rho'].mean(skipna=True)

            if not np.all(data.loc[id_day, 'T'].isnull()):
                temp[i] = data.loc[id_day, 'T'].mean(skipna=True)

            if not np.all(data.loc[id_day, 'sta_pa'].isnull()):
                pa[i] = data.loc[id_day, 'sta_pa'].mean(skipna=True)

    data_avg = pd.DataFrame({'days': day_list, 'V': vel, 'rho': rho, 'T': temp, 'pa': pa})
    return data_avg


def get_acf_opts(n_lags=75, do_sig=False, iters=1e3, alpha=1.0):
    """
    Function to provide a dictionary of parameters used in computing the acf of the in situ and remote sensing
    observations
    :param n_lags:
    :param do_sig:
    :param iters:
    :param alpha:
    :return:
    """
    acf_opts = {'n_lags': n_lags, 'do_sig': do_sig, 'iters': iters, 'alpha': alpha}
    return acf_opts

def compute_situ_acf(data, var, acf_opts=get_acf_opts()):
    """
    Function to compute the Spearman auto-correlation function (ACF) for a given series. Tailored to work with the daily
    mean dataframes used here. Processes the data as masked numpy arrays, using the masked stats in scipy stats, so that
    NaNs can be handled appropriately (not easy in pandas). Can also perform a significance test to assess the null
    hypothesis of zero autocorrelation. Significance test done by shuffling the series iters times, recomputing the
    ACF, and taking the percentiles of the shuffled acfs at alpha/2 and 1-(alpha/2). This can take hours to run, so
    reduce the iterations when testing figures.
    :param data: pandas dataframe containing the daily mean solar wind parameters
    :param var: string label of the key of the paramter to compute the acf for
    :param acf_opts: A dictionary containing some default paramters for computing the acf. Obtained form get_acf_opts().
                     To edit the params, make a call to get_acf_opts, edit the dict, then parse here.
    :return lags: Numpy array of lags.
    :return acf: Numpy array of the Spearman ACF .
    :return acf_sig (optional): Numpy array of the Spearman ACF giving the lower and upper significance levels of the
                                ACF under the null hypothesis the the ACF is zero at all lags.
    """
    #TODO add in some extra checks that the dictionary values are valid args.
    n_lags = np.int(acf_opts['n_lags'])
    do_sig = acf_opts['do_sig']
    iters = np.int(acf_opts['iters'])
    alpha = acf_opts['alpha']

    # pull out the data
    x = data[var].values
    # Make this a masked array
    x_bad = np.isnan(x)
    xm = np.ma.masked_where(x_bad, x)

    # Compute the ACF
    lags = np.arange(0, n_lags + 1.0, 1.0)
    acf = np.zeros(lags.size)
    acf_sig = np.zeros((2, lags.size))*np.NaN
    for i, l in enumerate(lags):
        if l == 0:
            acf[i] = st.mstats.spearmanr(xm, xm)[0]
        else:
            acf[i] = st.mstats.spearmanr(xm[:-i], xm[i:])[0]

    # Estimate the significance of the ACF compared to randomized realisations of the data.
    # Shuffle the data iters times, compute the acf for each iteration, bin it
    # Compute percentiles of the distribution of acf at each lag.
    if do_sig:
        acf_err = np.zeros((iters, lags.size))*np.NaN
        for i in range(iters):
            np.random.shuffle(xm)
            for j, l in enumerate(lags):
                # Skip first iteration as acf at lag=0 is by def 1, so error lims undefined.
                if l > 0:
                    acf_err[i, j] = st.mstats.spearmanr(xm[:-j], xm[j:])[0]

        # Calculate CI lims on null distribution of assumed independence.
        alpha2 = alpha / 2.0
        lims = [alpha2, 100.0 - alpha2]
        acf_sig = np.nanpercentile(acf_err, lims, axis=0)

    return lags, acf, acf_sig


def load_reduced_hi_data(t_start, t_stop, pa_lo=70.0, pa_hi=115, param='std'):
    """
    A function to load in a subset of the statistics computed on the HI frames. Pulls out the block of data between
    specified time and position angle limits.
    :param t_start: An astropy time object, specifying start of window
    :param t_stop: An astropy time object, specifying end of window
    :param pa_lo: int or float value of PA for bottom of pa window
    :param pa_hi: int or float value of PA for top of pa window
    :param param: string label of parameter to pull out of file. Set includes 'mean', 'median', 'std', and 'iqr'.
    :return:
    """

    if t_start > t_stop:
        print "Error: t_start must be earlier than t_stop."

    if not (isinstance(pa_lo, (int, long, float)) & isinstance(pa_hi, (int, long, float))):
        print "Error: pa_lo and pa_hi should be ints or floats.Resetting to defaults."
        pa_lo = 70.0
        pa_hi = 115.0

    if pa_lo > pa_hi:
        print "Error: pa_lo should be less than pa_hi. Resetting to defaults."
        pa_lo = 70.0
        pa_hi = 115.0

    pa_lims = np.array([pa_lo, pa_hi])
    if any(pa_lims < 70) | any(pa_lims > 115):
        print "Error: position angle limits should be between 70 and 115. Resetting to defaults."
        pa_lo = 70.0
        pa_hi = 115.0

    if param not in ['mean', 'median', 'std', 'iqr']:
        print "Error: param is not in list of available parametrs - [mean, median, std, iqr]."

    # Open the HDF5 of HI statistics
    proj_dirs = project_info()
    hi_data_path = proj_dirs['sta_hi_data']
    f = h5py.File(hi_data_path, "r")

    # Do a first pass to get the times and keys to get data subsets.
    keys = f.keys()
    key_times = np.zeros(len(keys))
    for i, k in enumerate(keys):
        if i == 0:
            pa_bins = f[k]['pa_bins']

        key_times[i] = f[k]['jd'].value

    # Put the times into an astropy time.
    key_times = Time(key_times, format='jd')

    # Now pull out the array of data matching the input arguments.
    # Find keys of this time window
    find_keys = np.argwhere((key_times.jd > t_start.jd) & (key_times.jd <= t_stop.jd)).ravel()
    # Get these times
    time = key_times[find_keys]
    # Also get the indices of the relevant pa bins
    id_pa = (pa_bins.value >= pa_lo) & (pa_bins.value <= pa_hi)
    pa_bins = pa_bins[id_pa]

    # Preallocate for retrieving data
    data = np.zeros((pa_bins.size, len(find_keys))) * np.NaN
    # Pull out this subset of the data
    for i, k in enumerate(find_keys):
        key = keys[k]
        data[:, i] = f[key][param][id_pa]

    # Close the HDF5 file.
    f.close()

    return time, pa_bins, data


def hi_daily_averaging(time, pa_bins, data, method="mean", reference_time="2008-01-01T00:00:00"):
    """
    A function to compute the daily average of the HI frame statistics. This is done in a fairly brute force way, to
    account for things like missing frames and NaN frames, which higher level routines (like pandas groupby) don't yet
    do well. So, this calculates the daily average for every day after the specified epoch date, using the specified
    averaging method. This returns a numpy masked array, with missing frames or full NaN frames masked.
    :param time:
    :param pa_bins:
    :param data:
    :param method:
    :return:
    """
    # Get times relative to epoch start
    t_start = Time(reference_time)
    frame_days = time - t_start  # a time delta
    # Get list of days for this whole window
    d_min = np.fix(frame_days.jd.min())
    d_max = np.fix(frame_days.jd.max())
    day_list = TimeDelta(np.arange(d_min, d_max + 1.0, 1.0), format='jd')
    dt = TimeDelta(0.5, format='jd')
    # Make space to store the averages and uncertaintites
    data_avg = np.zeros((pa_bins.size, day_list.size)) * np.NaN

    for i, day in enumerate(day_list):

        id_day = (frame_days.jd > (day - dt).jd) & (frame_days.jd <= (day + dt).jd)

        # If some entries found for this day, compute the mean
        if np.any(id_day):

            # Check not all data is invalid
            if not np.all(np.isnan(data[:, id_day])):

                # Some data is good, so compute average with specified method.
                if method == "mean":
                    for j in range(pa_bins.size):
                        data_avg[j, i] = np.nanmean(data[j, id_day])
                elif method == "median":
                    for j in range(pa_bins.size):
                        data_avg[j, i] = np.nanmedian(data[j, id_day])
                else:
                    print "Invalid method specified. Returning NaN array"
                    data_avg[:, i] = np.NaN

    # Make masked arrays for output, masking bad values in average AND/OR error.
    id_bad = np.isnan(data_avg)
    num_bad = np.sum(id_bad)
    num_good = data_avg.size - num_bad
    print "Averaging complete: {} bad samples of {} total.".format(num_bad, num_good)
    data_avg = np.ma.masked_where(id_bad, data_avg)
    data_avg.set_fill_value(np.NaN)

    return day_list, data_avg


def compute_hi_acf(pa_bins, data, acf_opts=get_acf_opts()):
    """
    Function to compute the Spearman ACF of the HI variability data
    :param pa_bins:
    :param data:
    :param acf_opts:
    :return:
    """
    # make sure iters is an int, as it is also used as an index.
    n_lags = np.int(acf_opts['n_lags'])
    do_sig = acf_opts['do_sig']
    iters = np.int(acf_opts['iters'])
    alpha = acf_opts['alpha']

    # Preallocate space for acf.
    lags = np.arange(0, n_lags + 1.0, 1.0)
    acf = np.zeros((pa_bins.size, lags.size)) * np.NaN
    acf_sig = np.zeros((pa_bins.size, lags.size, 2)) * np.NaN
    # Compute the ACF
    for i in range(pa_bins.size):

        for j, l in enumerate(lags):
            if l == 0:
                acf[i, j] = st.mstats.spearmanr(data[i, :], data[i, :])[0]
            else:
                acf[i, j] = st.mstats.spearmanr(data[i, :-j], data[i, j:])[0]

        # Estimate the significance of the ACF compared to randomized realisations of the data.
        # Shuffle the data iters times, compute the acf for each iteration, bin it
        # Compute percentiles of the distribution of acf at each lag.
        if do_sig:
            # Grab a copy of this pa row for shuffling
            data_sub = data[i, :].copy()
            # Space for the replicants on this iter
            acf_err = np.zeros((iters, lags.size)) * np.NaN
            # Loop over the replications + shuffle
            for k in range(iters):
                np.random.shuffle(data_sub)
                # Compute acfs of randomised replicant
                for j, l in enumerate(lags):
                    if l > 0:
                        acf_err[k, j] = st.mstats.spearmanr(data_sub[:-j], data_sub[j:])[0]

            # Calculate CI lims on null distribution of assumed independence.
            alpha2 = alpha / 2.0
            lims = [alpha2, 100.0 - alpha2]
            acf_sig[i, :, :] = np.nanpercentile(acf_err, lims, axis=0).T

    return lags, acf, acf_sig
