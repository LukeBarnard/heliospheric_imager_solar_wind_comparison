import numpy as np
import scipy.stats as st
import pandas as pd
import os
import glob
import pickle
# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
# Space science packages
from astropy.time import Time
import astropy.units as u
import sunpy.sun.constants as solarconst
import sunpy.sun as sun
# My own packages used for this work
from stereo_spice.coordinates import StereoSpice
import heliospheric_imager_analysis.images as hip
import solarwind_processing as swp

# Get stereo_spice instance for later.
spice = StereoSpice()

# Configure defaults for matplotlib
mpl.rc("axes", labelsize=14)
mpl.rc("ytick", labelsize=13)
mpl.rc("xtick", labelsize=13)
mpl.rc("legend", fontsize=13)

def import_solar_wind_plasma_data(t_start="2008-01-01T00:00:00", t_stop="2012-12-31T23:59:59"):
    """
    Load in the various raw solar wind plasma data, clean up the data and make it consistent, having the
    same format and span. This is then saved as a CSV for quick loading into PANDAS in the following analysis.
    :param t_start: timestring of the beginning time of the observation period to return
    :param t_stop: timestring of the end time of the observation period to return
    :return:
    """
    fmt = "%Y-%m-%dT%H:%M:%S"
    t_start = pd.datetime.strptime(t_start, fmt)
    t_stop = pd.datetime.strptime(t_stop, fmt)

    for src in ['wind', 'sta', 'stb']:

        if src == 'wind':
            data = swp.import_wind_data()
        elif src == 'sta':
            data = swp.import_sta_data()
        elif src == 'stb':
            data = swp.import_stb_data()

        # Restrict to window of interest
        find_period = (data['time'] >= t_start) & (data['time'] <= t_stop)
        data = data.loc[find_period, :]

        # Save to csv.
        proj_dirs = swp.project_info()
        out_name = "{}_mapped_speed_data_{}_{}.csv".format(src, t_start.strftime("%Y%m%d"), t_stop.strftime("%Y%m%d"))
        out_path = os.path.join(proj_dirs['data'], out_name)
        data.to_csv(out_path)
    return


def insitu_time_series_acf():
    """
    Compute the autocorrelation function of the in-situ solar wind parameters, and plot these with the time series, for
    each of STEREO-A, STEREO-B and WIND.
    :return:
    """
    acf_opts = swp.get_acf_opts()
    acf_opts['do_sig'] = True
    acf_opts['iters'] = 1000

    fig, ax, axl, axr = setup_time_series_acf_figure()
    craft_cols = swp.get_craft_colors()

    # Now load in the in-situ data. take daily means. compute spearman acf and sig. plot.
    epoch_start = "2008-01-01T00:00:00"
    t_start = Time(epoch_start)
    for sw_param in ['V', 'rho', 'T']:

        if sw_param == "V":
            ylabel_unit = r"Plasma speed ($km/s$)"
            ylim = [250, 800]
        elif sw_param == "rho":
            ylabel_unit = r"Density ($cm^{{-3}}$)"
            ylim = [0, 20]
        elif sw_param == "T":
            ylabel_unit = r"Temperature ($MK$)"
            ylim = [0, 0.5]

        for i, craft in enumerate(['wind', 'stb', 'sta']):

            if craft == 'wind':
                situ = swp.load_wind_data()
            elif craft == 'stb':
                situ = swp.load_stb_data()
            elif craft == 'sta':
                situ = swp.load_sta_data()

            situ_avg = swp.situ_daily_averaging(situ, reference_time=epoch_start)

            axl[i].plot(situ_avg['days'], situ_avg[sw_param], '-', color=craft_cols[i])
            axl[i].set_ylabel(r"{} {}".format(craft.upper(), ylabel_unit))

            # Now compute acf and plot.
            lags, acf, acf_sig = swp.compute_situ_acf(situ_avg, sw_param, acf_opts=acf_opts)

            axr[i].plot(lags, acf, 'k-')
            axr[i].plot(lags, acf_sig[0, :], 'r--')
            axr[i].plot(lags, acf_sig[1, :], 'r--')

        for al, ar in zip(axl[:2], axr[:2]):
            al.set_xticklabels([])
            ar.set_xticklabels([])

        axl[-1].set_xlabel("Days since {}".format(t_start.datetime.strftime("%Y-%m-%d")))
        axr[-1].set_xlabel("Lag (days)")

        for i, (al, ar) in enumerate(zip(axl, axr)):
            al.set_xlim(situ_avg['days'].min(), situ_avg['days'].max())
            al.set_ylim(ylim[0], ylim[1])

            ar.set_xlim(0, lags.max())
            ar.set_ylim(-0.15, 1.0)

            label = "A{})".format(i + 1)
            al.text(0.01, 0.92, label, color='k', transform=al.transAxes)

            label = "B{})".format(i + 1)
            ar.text(0.0275, 0.92, label, color='k', transform=ar.transAxes)

            al.tick_params('both')
            ar.tick_params('both')

        for ar in axr:
            ar.set_ylabel('Spearman correlation')
            ar.yaxis.tick_right()
            ar.yaxis.set_label_position("right")

        fig.subplots_adjust(left=0.05, bottom=0.06, right=0.95, top=0.99, wspace=0.03, hspace=0.04)
        proj_dirs = swp.project_info()
        fig_name = "{}_time_series_and_acf.png".format(sw_param)
        fig_path = os.path.join(proj_dirs['figs'], fig_name)
        fig.savefig(fig_path)

        for a in ax:
            a.clear()

    return


def HI_time_series_acf():
    """
    Plot the time-series of HI1A variability, and the acfs, for all available position angles, and along 2 fixed
    position angles.
    """

    proj_dirs = swp.project_info()
    reduced_hi_data_path = os.path.join(proj_dirs['data'], "reduced_hi_data_std.pickle")
    with open(reduced_hi_data_path, "rb") as f:
        from_pickle = pickle.load(f)

    time = from_pickle['time'].copy()
    pa_bins = from_pickle['pa_bins'].copy()
    data = from_pickle['data'].copy()
    del from_pickle

    ts = Time("2008-01-01T00:00:00")
    data_z = (data - np.nanmean(data, axis=0)) / np.nanstd(data, axis=0)
    time_avg, data_avg = swp.hi_daily_averaging(time, pa_bins, data_z, method="mean")

    acf_opts = swp.get_acf_opts()
    acf_opts['do_sig'] = True
    acf_opts['iters'] = 1000

    fig, ax, axl, axr = setup_time_series_acf_figure()
    pa_cols = swp.get_pa_colors()

    # Plot out variability for all position angles
    lims = [-1.5, 1.5]
    hi_cmap = plt.cm.PiYG
    hi_cmap.set_bad('k')
    hi_norm = plt.Normalize(vmin=lims[0], vmax=lims[1])
    hi_cax = axl[0].pcolormesh(time_avg.jd, pa_bins, data_avg, norm=hi_norm, cmap=hi_cmap)
    # Compute ACF of this.
    lags, acf, acf_sig = swp.compute_hi_acf(pa_bins, data_avg, acf_opts=acf_opts)
    # Contour the ACF.
    cor_cmap = plt.cm.coolwarm
    cor_norm = plt.Normalize(vmin=-1.0, vmax=1.0)
    dl = 0.1
    levels = np.arange(-0.9, 0.9 + dl, dl)
    cor_cax = axr[0].contourf(lags, pa_bins, acf, norm=cor_norm, cmap=cor_cmap, levels=levels)
    # Add hatching of significance
    find_sig = ~((acf > acf_sig[:, :, 0]) & (acf < acf_sig[:, :, 1]))
    hatching = np.ma.masked_where(find_sig, acf)
    axr[0].pcolor(lags, pa_bins, hatching, hatch='x', alpha=0.0)

    # Plot out variability along a fixed position angle
    pa_fix = 90.0
    axl[0].plot(time_avg.jd, pa_fix + np.zeros(time_avg.size), '--', color=pa_cols[0], linewidth=2)
    id_pa = np.argmin(np.abs(pa_bins - pa_fix))
    data_avg_fix_pa = data_avg[id_pa, :].copy()
    axl[1].plot(time_avg.jd, data_avg_fix_pa, '-', color=pa_cols[0])
    axr[1].plot(lags, acf[id_pa, :], 'k-')
    axr[1].plot(lags, acf_sig[id_pa, :], 'r--')
    axr[1].plot(lags, acf_sig[id_pa, :], 'r--')

    # Now plot out variability along a PA track similar to a in-situ monitor.
    situ = swp.load_wind_data()
    situ_avg = swp.situ_daily_averaging(situ)
    axl[0].plot(situ_avg['days'], situ_avg['pa'], '-.', color=pa_cols[1], linewidth=2)
    # compute variability along this track.
    var_track = np.zeros(situ_avg.shape[0]) * np.NaN
    for i, row in situ_avg.iterrows():
        id_day = np.where(time_avg.jd == row['days'])[0]
        id_pa = np.argmin(np.abs(pa_bins - row['pa']))
        if id_day.size != 0:
            var_track[i] = data_avg[id_pa, id_day]

    axl[2].plot(situ_avg['days'], var_track, '-', color=pa_cols[1])
    # Compute acf of the track variability.
    situ_avg['hi_var'] = var_track
    lags_track, acf_track, acf_track_sig = swp.compute_situ_acf(situ_avg, 'hi_var', acf_opts=acf_opts)
    axr[2].plot(lags_track, acf_track, 'k-')
    axr[2].plot(lags_track, acf_track_sig[0], 'r--')
    axr[2].plot(lags_track, acf_track_sig[1], 'r--')

    for al, ar in zip(axl[:2], axr[:2]):
        al.set_xticklabels([])
        ar.set_xticklabels([])

    axl[-1].set_xlabel("Days since {}".format(ts.datetime.strftime("%Y-%m-%d")))
    axr[-1].set_xlabel("Lag (days)")

    for a in [axl[0], axr[0]]:
        a.set_ylabel("Position angle (degrees)")
        a.set_ylim(pa_bins.min(), pa_bins.max())

    for al in axl[1:]:
        al.set_ylabel("HI1-A Variability")

    for ar in axr[1:]:
        ar.set_ylabel("Spearman correlation")
        ar.set_ylim(-0.15, 1.0)

    for i, (al, ar) in enumerate(zip(axl, axr)):
        al.set_xlim(time_avg.jd.min(), time_avg.jd.max())
        ar.set_xlim(0, lags.max())

        label = "A{})".format(i + 1)
        al.text(0.01, 0.9, label, backgroundcolor="white", color='k', transform=al.transAxes)

        label = "B{})".format(i + 1)
        ar.text(0.0275, 0.9, label, backgroundcolor="white", color='k', transform=ar.transAxes)

    for ar in axr:
        ar.yaxis.tick_right()
        ar.yaxis.set_label_position("right")

    fig.subplots_adjust(left=0.05, bottom=0.06, right=0.95, top=0.925, wspace=0.03, hspace=0.04)

    # Add in the colorbar
    pos = axl[0].get_position()
    dw = 0.005
    dh = 0.005
    left = pos.x0 + dw
    bottom = pos.y1 + dh
    wid = pos.width - 2 * dw
    hi_cbaxes = fig.add_axes([left, bottom, wid, 0.015])
    cbar1 = fig.colorbar(hi_cax, cax=hi_cbaxes, orientation='horizontal')
    cbar1.ax.set_xlabel("HI1-A Variability")
    cbar1.ax.xaxis.tick_top()
    cbar1.ax.xaxis.set_label_position('top')

    pos = axr[0].get_position()
    dw = 0.005
    dh = 0.005
    left = pos.x0 + dw
    bottom = pos.y1 + dh
    wid = pos.width - 2 * dw

    cor_cbaxes = fig.add_axes([left, bottom, wid, 0.015])
    cbar2 = fig.colorbar(cor_cax, cax=cor_cbaxes, orientation='horizontal')
    cbar2.set_ticks([-0.8, -0.4, 0.0, 0.4, 0.8])
    cbar2.ax.set_xlabel("Spearman correlation")
    cbar2.ax.xaxis.tick_top()
    cbar2.ax.xaxis.set_label_position('top')

    proj_dirs = swp.project_info()
    out_path = os.path.join(proj_dirs['figs'], 'remote_sensing_time_series_and_acf.png')
    fig.savefig(out_path)
    return


def rolling_acf():
    """
    Compute the rolling acf of the HI1A variability data and the in-situ plasma observations.
    :return:
    """
    proj_dirs = swp.project_info()
    reduced_hi_data_path = os.path.join(proj_dirs['data'], "reduced_hi_data_std.pickle")
    with open(reduced_hi_data_path, "rb") as f:
        from_pickle = pickle.load(f)

    remote_time = from_pickle['time'].copy()
    pa_bins = from_pickle['pa_bins'].copy()
    remote = from_pickle['data'].copy()
    del from_pickle

    ts = Time("2008-01-01T00:00:00")
    remote_z = (remote - np.nanmean(remote, axis=0))/np.nanstd(remote, axis=0)
    remote_time_avg, remote_avg = swp.hi_daily_averaging(remote_time, pa_bins, remote_z, method="mean",
                                                         reference_time=ts.isot)
    del remote, remote_time

    # Compute rolling acf of in-situ params and HI variability
    fig, ax = plt.subplots(2, 3, figsize=(14, 9.5))
    ax_wnd = ax[:, 0]
    ax_sta = ax[:, 1]
    ax_stb = ax[:, 2]
    ax_situ = ax[0, :]
    ax_remote = ax[1, :]

    # Setup autocorrelation options
    acf_opts = swp.get_acf_opts()
    acf_opts['do_sig'] = True
    acf_opts['iters'] = 1000
    # Arguments for the rolling autocorrelation window
    acf_window_wid = 365
    acf_window_step = 14

    # Set up the colormap of the ACF contours and the normalisation. Same for each panel.
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=-1.0, vmax=1.0)
    dl = 0.1
    levels = np.arange(-0.9, 0.9 + dl, dl)

    for src, ax_cft in zip(['wind', 'sta', 'stb'], [ax_wnd, ax_sta, ax_stb]):

        # Load in the in-situ for this observatory
        if src == 'wind':
            situ = swp.load_wind_data()
        elif src == 'sta':
            situ = swp.load_sta_data()
        elif src == 'stb':
            situ = swp.load_stb_data()

        # Compute daily mean with same reference time as remote sensing obs.
        situ_avg = swp.situ_daily_averaging(situ, reference_time=ts.isot)
        # Occasionally a missing value in PA. replace with interpolted est - smooth function so good approx.
        num_bad = np.sum(np.isnan(situ_avg['pa']))
        if num_bad > 0:
            situ_avg['pa'] = situ_avg['pa'].interpolate()
            print "Notice: {} bad PA values found. Replacing with interpolates".format(num_bad)

        # Directly lookup the variability along the pa path.
        pa_track = np.zeros(situ_avg.shape[0]) * np.NaN
        var_track = np.zeros(situ_avg.shape[0]) * np.NaN
        for i, row in situ_avg.iterrows():
            id_day = np.where(remote_time_avg.jd == row['days'])[0]
            id_pa = np.argmin(np.abs(pa_bins - row['pa']))
            pa_track[i] = pa_bins[id_pa]
            if id_day.size != 0:
                var_track[i] = remote_avg[id_pa, id_day]
        situ_avg['pa_track'] = pa_track
        situ_avg['var_track'] = var_track

        # Get final block start time for this window length
        t_max = situ_avg.shape[0] - acf_window_wid
        # Get array of start times for this window and block_step.
        t_start = np.arange(0, t_max, acf_window_step, dtype=np.int)
        # Get array of lags used in ACF (for preallocating space)...
        lags = np.arange(0, acf_opts['n_lags'] + 1.0, 1.0)

        # Loop through the parameters
        for j, var in enumerate(['V', 'var_track']):

            # Update where we are at, as this takes an age.
            print("Computing rolling acf for {} - {}".format(src.upper(), var))

            # preallocate space for the stack of ACFs and significance calcs.
            acf_stack = np.zeros((lags.size, t_start.size))
            sig_stack = np.zeros((lags.size, t_start.size, 2))

            # Loop through the block start times
            for k, t in enumerate(t_start):

                # Pull out this block of data
                data_block = situ_avg.loc[t: t + acf_window_wid].copy()
                # Compute ACF and significance, stash into the stack
                lags, acf, acf_sig = swp.compute_situ_acf(data_block, var, acf_opts=acf_opts)
                acf_stack[:, k] = acf
                sig_stack[:, k, 0] = acf_sig[0]
                sig_stack[:, k, 1] = acf_sig[1]

            # contour the rolling acf functions
            cntr_h = ax_cft[j].contourf(t_start, lags, acf_stack, norm=norm, cmap=cmap, levels=levels)
            # add hatching for significance
            find_sig = ~((acf_stack > sig_stack[:, :, 0]) & (acf_stack < sig_stack[:, :, 1]))
            sig = np.ma.masked_where(find_sig, acf_stack)
            ax_cft[j].pcolor(t_start, lags, sig, hatch='x', alpha=0.0)

    # Format the axes
    for a in ax.ravel():
        a.set_xlim(t_start.min(), t_start.max())
        a.set_ylim(lags.min(), lags.max())

    for src, ax_cft in zip(['wind', 'sta', 'stb'], [ax_wnd, ax_sta, ax_stb]):
        ax_cft[-1].set_xlabel("Block start (days since {})".format(ts.datetime.strftime("%Y-%m-%d")))
        for a in ax_cft[:1]:
            a.set_xticklabels([])

    for param, ax_prm in zip(['V', 'HI. Var.'], [ax_situ, ax_remote]):
        ax_prm[0].set_ylabel("Lag (days)")
        for a in ax_prm[1:]:
            a.set_yticklabels([])

    for i, (letter, param) in enumerate(zip(['A', 'B'], ['V', 'HI var.'])):
        for j, crft in enumerate(['WIND', 'STA', 'STB']):
            label = "{}{}) {}-{}".format(letter, j, crft, param)
            ax[i, j].text(0.0275, 0.92, label, backgroundcolor='whitesmoke', color='k', transform=ax[i, j].transAxes)

    # Adjust axes position and add a colorbar giving scale of the ACFs
    fig.subplots_adjust(left=0.055, bottom=0.055, right=0.98, top=0.915, hspace=0.03, wspace=0.03)
    cax = fig.add_axes([0.055, 0.92, 0.925, 0.025])
    cax2 = fig.colorbar(cntr_h, cax, orientation='horizontal')
    cax2.ax.xaxis.tick_top()
    cax2.ax.set_xlabel('Spearman Correlaton')
    cax2.ax.xaxis.set_label_position('top')

    # Save the figure
    proj_dirs = swp.project_info()
    out_name = "rolling_acf_new.png"
    out_path = os.path.join(proj_dirs['figs'], out_name)
    fig.savefig(out_path)
    return


def lagged_correlation():
    """
    Compute the lagged correlation between HI1A variability and in-situ solar wind speeds, mapped to the position angle
    of their approximate source location from backward propagating the in-situ solar wind observations.
    :return:
    """
    proj_dirs = swp.project_info()
    reduced_hi_data_path = os.path.join(proj_dirs['data'], "reduced_hi_data_std.pickle")
    with open(reduced_hi_data_path, "rb") as f:
        from_pickle = pickle.load(f)

    remote_time = from_pickle['time'].copy()
    pa_bins = from_pickle['pa_bins'].copy()
    remote = from_pickle['data'].copy()
    del from_pickle

    ts = Time("2008-01-01T00:00:00")
    remote_z = (remote - np.nanmean(remote, axis=0)) / np.nanstd(remote, axis=0)
    remote_time_avg, remote_avg = swp.hi_daily_averaging(remote_time, pa_bins, remote_z, method="mean",
                                                         reference_time=ts.isot)
    del remote, remote_time

    # Compute rolling acf of in-situ params and HI variability
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    lagged_corr_colors = swp.get_lagged_corr_colors()
    years = np.arange(0, 5 * 365, 365)
    yr_cols = [lagged_corr_colors.mpl_colors[i] for i in [5, 4, 3, 2, 1]]

    # Params for significance calc.
    n_resamples = 1000
    alpha = 1.0
    alpha2 = alpha / 2.0
    lims = [alpha2, 100.0 - alpha2]

    lag_corr_stats = np.zeros((years.size, 3, 3))

    for cc, src in enumerate(['wind', 'sta', 'stb']):

        # Load in the in-situ for this observatory
        if src == 'wind':
            situ = swp.load_wind_data()
        elif src == 'sta':
            situ = swp.load_sta_data()
        elif src == 'stb':
            situ = swp.load_stb_data()

        # Compute daily mean with same reference time as remote sensing obs.
        situ_avg = swp.situ_daily_averaging(situ, reference_time=ts.isot)
        # Occasionally a missing value in PA. replace with interpolted est - smooth function so good approx.
        num_bad = np.sum(np.isnan(situ_avg['pa']))
        if num_bad > 0:
            situ_avg['pa'] = situ_avg['pa'].interpolate()
            print "Notice: {} bad PA values found. Replacing with interpolates".format(num_bad)

        year_name = [2008, 2009, 2010, 2011, 2012]

        for yy, year in enumerate(years):
            id_situ = (situ_avg['days'] >= year) & (situ_avg['days'] <= year + 365)
            situ_sub = situ_avg.loc[id_situ, :].copy()

            id_remote = (remote_time_avg.jd >= year) & (remote_time_avg.jd <= year + 365)
            remote_time_sub = remote_time_avg[id_remote].copy()
            remote_sub = remote_avg[:, id_remote].copy()

            # Directly lookup the variability along the pa path.
            lags = np.arange(-27, 28, 1, dtype=np.int)
            corr = np.zeros(lags.shape) * np.NaN
            sig_corr = np.zeros((lags.size, 2))

            for k, lag in enumerate(lags):

                pa_track = np.zeros(situ_sub.shape[0]) * np.NaN
                var_track = np.zeros(situ_sub.shape[0]) * np.NaN
                iii = 0
                for i, row in situ_sub.iterrows():

                    id_day = np.where(remote_time_sub.jd == row['days'] + lag)[0]
                    id_pa = np.argmin(np.abs(pa_bins - row['pa']))
                    pa_track[iii] = pa_bins[id_pa]
                    if id_day.size != 0:
                        var_track[iii] = remote_sub[id_pa, id_day]
                    iii += 1

                situ_sub['pa_track'] = pa_track
                situ_sub['var_track'] = var_track

                # Compute spearman r and resampling significance.
                corr[k] = st.mstats.spearmanr(situ_sub['V'], situ_sub['var_track'])[0]

                x = situ_sub['V'].values.copy()
                y = situ_sub['var_track'].values.copy()
                resamples = np.zeros(n_resamples) * np.NaN
                for kk in range(resamples.size):
                    resamples[kk] = st.mstats.spearmanr(x, y)[0]
                    np.random.shuffle(x)
                    np.random.shuffle(y)

                # Get percentiles of the null under resampling independence
                null_lims = np.nanpercentile(resamples, lims)
                sig_corr[k, :] = null_lims

            lag_corr_stats[yy, 0, ] = year
            if (src == "wind") | (src == 'sta'):

                id_lo = lags < 0
                lags2 = lags[id_lo]
                corr2 = corr[id_lo]
                id_min = np.argmin(corr2)
                lag_corr_stats[yy, 1, cc] = lags2[id_min]
                lag_corr_stats[yy, 2, cc] = corr2[id_min]
            elif src == 'stb':
                id_mid = (lags > -10) & (lags < 10)
                lags2 = lags[id_mid]
                corr2 = corr[id_mid]
                id_min = np.argmin(corr2)
                lag_corr_stats[yy, 1, cc] = lags2[id_min]
                lag_corr_stats[yy, 2, cc] = corr2[id_min]

            # Update the plot with this years data.
            ax[cc].plot(lags, corr, 'o-', color=yr_cols[yy], markerfacecolor='None', label="{}".format(year_name[yy]))
            id_sig = (corr < sig_corr[:, 0]) | (corr > sig_corr[:, 1])
            ax[cc].plot(lags[id_sig], corr[id_sig], 'o', color=yr_cols[yy])

        ax[cc].text(0.0275, 0.92, src.upper(), color='k', transform=ax[cc].transAxes, fontsize=14)
        ax[cc].set_xlabel("Lag (days)", fontsize=14)

    for a in ax:
        a.legend(fontsize=13, ncol=2)
        a.set_xlim(lags.min(), lags.max())
        a.set_ylim(-0.55, 0.5)
        a.set_xticks(np.arange(-25, 30, 5))
        a.tick_params('both', labelsize=14)
    for a in ax[1:]:
        a.set_yticklabels([])

    ax[0].set_ylabel('Spearman correlation', fontsize=14)

    fig.subplots_adjust(left=0.075, bottom=0.11, right=0.98, top=0.98, wspace=0.025)
    proj_dirs = swp.project_info()
    out_name = "hi_var_situ_lagged_corr.png"
    out_path = os.path.join(proj_dirs['figs'], out_name)
    fig.savefig(out_path)

    # Scatter plot of lag vs yr
    fig, ax = plt.subplots(figsize=(7, 6))
    craft_cols = swp.get_craft_colors()
    craft_mkr = ['o', 's', '^']

    years = (np.arange(2008, 2013, 1)) + 0.5 - 2008

    for i, craft in enumerate(['WIND', 'STA', 'STB']):
        x = lag_corr_stats[:, 0, i]
        y = lag_corr_stats[:, 1, i]
        ax.plot(years, y, linestyle='None', marker=craft_mkr[i], color=craft_cols[i], label=craft.upper())
        slope, intercept, r_value, p_value, std_err = st.linregress(years, y)
        print "{}: Gradient: {} +/- {}".format(craft, slope, std_err)
        ax.plot(years, intercept + slope * years, '-', color=craft_cols[i])
        ax.plot(years, intercept + (slope - 2 * std_err) * years, '--', color=craft_cols[i])
        ax.plot(years, intercept + (slope + 2 * std_err) * years, '--', color=craft_cols[i])
        # Add in another case to exclude outlying STA point in 2012.
        if craft == 'STA':
            years_sub = years[:-1]
            y_sub = y[:-1]
            slope, intercept, r_value, p_value, std_err = st.linregress(years_sub, y_sub)
            print "{}: Gradient (no outlier): {} +/- {}".format(craft, slope, std_err)
            ax.plot(years_sub, intercept + slope * years_sub, ':', color=craft_cols[i])
            ax.plot(years_sub, intercept + (slope - 2 * std_err) * years_sub, '-.', color=craft_cols[i])
            ax.plot(years_sub, intercept + (slope + 2 * std_err) * years_sub, '-.', color=craft_cols[i])

    ax.legend(fontsize=14)
    ax.set_xlabel('Years since 2008-01-01', fontsize=14)
    ax.set_ylabel('Lag of peak correlation (days)', fontsize=14)
    ax.tick_params('both', labelsize=14)
    ax.set_ylim(-27, 10)
    ax.set_xlim(0.45, 4.55)
    fig.subplots_adjust(left=0.11, bottom=0.1, right=0.99, top=0.99)
    proj_dirs = swp.project_info()
    out_name = "peak_lag_with_time.png"
    out_path = os.path.join(proj_dirs['figs'], out_name)
    fig.savefig(out_path)
    return


def get_circ_coords(xcenter, ycenter, radius):
    """
    Function to return the cartesian coordinates of the perimeter of a circle, at each degree of the circles perimeter.
    :param xcenter: x-coordinate of the circles center.  Unit must be consistent with ycenter and radius.
    :param ycenter: y-coordinate of the circles center. Unit must be consistent with xcenter and radius.
    :param radius: radius of the circle. Unit must be consistent with xcenter and ycenter.
    :return:
    """
    width, height = 2*radius, 2*radius
    angle = 0.0

    theta = np.deg2rad(np.arange(0.0, 360.0, 1.0))
    x = 0.5 * width * np.cos(theta)
    y = 0.5 * height * np.sin(theta)

    rtheta = np.radians(angle)
    R = np.array([
        [np.cos(rtheta), -np.sin(rtheta)],
        [np.sin(rtheta),  np.cos(rtheta)],
        ])

    x, y = np.dot(R, np.array([x, y]))
    x += xcenter
    y += ycenter
    return x, y


def get_parker_spiral(ro, r_fin, phi_fin, v):
    """
    Function to compute the cartesian coordinates of an idealised Parker spiral streamline in the ecliptic plane,
    between two radial coordinates, for every 0.01 radians of longitude.
    :param ro: Radial coordinate of the start point of the streamline (in solar radii)
    :param r_fin: Radial coordinate of the end point of the streamline (in solar radii)
    :param phi_fin: Longitudinal coordinate of the end point of the streamline (in radians)
    :param V: Assumed value for constant solar wind speed (in km/s)
    :return:
    """
    rs = solarconst.radius.to('km').value
    psi = 2.7e-6*(u.rad/u.s)
    v = v * (u.km/u.s) / rs
    phio = phi_fin - (r_fin - ro)*(psi / v).value
    phi = np.arange(phio, phi_fin, 0.01)
    r = ro + (v / psi).value*(phi - phio)
    x = r * np.sin(phi)
    y = r * np.cos(phi)
    return x, y


def hi1a_thomson_sphere_and_parker_spiral():
    """
    Produce a plot showing the approximate layout of the Thomson sphere of HI1A, including the idealised Parker spiral
    streamlines connecting STEREO-A, STEREO-B, and WIND to the corona.
    """
    craft_cols = swp.get_craft_colors()

    rs = solarconst.radius.to('km').value

    # Plot each event individually
    time = ["{}-06-15".format(i) for i in range(2008,2013)]
    time = Time(time)

    fig, ax = plt.subplots(2, len(time), figsize=(15, 6))
    ax_t = ax[0, :]
    ax_b = ax[1, :]

    for t, at, ab in zip(time, ax_t, ax_b):

        system = 'HEEQ'
        wnd = swp.get_wind_coords(t, system)
        sta = spice.get_coord(t, 'sta', system,  no_velocity=True)
        stb = spice.get_coord(t, 'stb', system, no_velocity=True)

        wnd = wnd / rs
        sta = sta / rs
        stb = stb / rs

        sta_r = np.sqrt(np.sum(sta**2))
        wnd_r = np.sqrt(np.sum(wnd**2))
        stb_r = np.sqrt(np.sum(stb**2))

        for a in [at, ab]:
            # Plot the sun, and source elongations considered
            rad = rs / rs
            xcenter, ycenter = 0, 0
            x, y = get_circ_coords(xcenter, ycenter, rad)
            a.fill(x, y, facecolor='orange', edgecolor='orange', linewidth=2, zorder=1)

            rad = (20.0 * rs / rs)
            x, y = get_circ_coords(xcenter, ycenter, rad)
            a.fill(x, y, facecolor='None', edgecolor='k', linewidth=2, zorder=1)

            rad = (22.5 * rs / rs)
            x, y = get_circ_coords(xcenter, ycenter, rad)

            a.fill(x, y, facecolor='None', edgecolor='k', linewidth=2, zorder=1)

            # Plot the TS for craft at x=au, y=0
            rad = sta_r / 2.0
            xcenter, ycenter = sta[1] / 2.0, sta[0] / 2.0
            x, y = get_circ_coords(xcenter, ycenter, rad)
            a.fill(x, y, facecolor='None', edgecolor=craft_cols[1], linestyle='--', linewidth=2, zorder=1)

            # Add on the craft
            a.plot(sta[1], sta[0], 's', color=craft_cols[1], label='STA')

            a.plot(wnd[1], wnd[0], 'o', color=craft_cols[0], label='WIND')

            a.plot(stb[1], stb[0], '^', color=craft_cols[2], label='STB')

            # Add on idealised Parker Spiral for STEREO-A, STEREO-B and WIND
            ro = 22.5
            V = 400.0
            phi_fin = np.arctan2(wnd[0], wnd[1])
            x, y = get_parker_spiral(ro, wnd_r, phi_fin, V)
            a.plot(y, x, '-', color=craft_cols[0])

            phi_fin = np.arctan2(sta[0], sta[1])
            x, y = get_parker_spiral(ro, sta_r, phi_fin, V)
            a.plot(y, x, '-', color=craft_cols[1])

            phi_fin = np.arctan2(stb[0], stb[1])
            x, y = get_parker_spiral(ro, stb_r, phi_fin, V)
            a.plot(y, x, '-', color=craft_cols[2])

        label = t.datetime.strftime("%Y-%m-%d")
        at.text(0.6, 0.925, label, color='k', transform=at.transAxes, fontsize=14)
        at.legend(loc=2)

    for a in ax_t:
        a.set_xlim(-225, 225)
        a.set_ylim(-225, 225)

    for a in ax_b:
        a.set_xlim(-50, 50)
        a.set_ylim(-50, 50)

    for a in ax.ravel():
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.set_xticks([])
        a.set_yticks([])
        a.set_aspect('equal')
        a.invert_yaxis()

    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0, hspace=0)
    proj_dirs = swp.project_info()
    out_name = "hi1a_ts_relative_locations.png"
    out_path = os.path.join(proj_dirs['figs'], out_name)
    fig.savefig(out_path)
    return


def spacecraft_orbits():
    """
    Produce a plot of the positions of STEREO-A, STEREO-B and WIND in Heliospheric Equatorial Coordinates
    over the studied period 2008-01-01 to 2012-12-31.
    """
    craft_cols = swp.get_craft_colors()

    au = solarconst.au.to('km').value

    # Plot each event individually
    t_s = pd.date_range("2008-01-01", "2013-01-01", freq="D")
    time = Time(t_s.to_pydatetime())

    system = 'HEEQ'
    wnd = swp.get_wind_lonlat(time, system)
    sta = spice.get_lonlat(time, 'sta', system, degrees=True)
    stb = spice.get_lonlat(time, 'stb', system, degrees=True)

    wnd[:, 0] = wnd[:, 0] / au
    sta[:, 0] = sta[:, 0] / au
    stb[:, 0] = stb[:, 0] / au

    fig, ax = plt.subplots(3, 1, figsize=(14, 7))

    for data, col, label in zip([wnd, sta, stb], craft_cols, ['WIND', 'STA', 'STB']):
        for i, a in enumerate(ax):
            a.plot(time.to_datetime(), data[:, i], '-', color=col, label=label)

    for a in ax:
        a.legend(frameon=False)
        a.set_xlim(time.to_datetime().min(), time.to_datetime().max())

    for a in ax[0:2]:
        a.set_xticklabels([])

    ymin = 0.94
    ymax = 1.1
    ax[0].set_ylim(ymin, ymax)

    fnt = 15
    ax[0].set_ylabel('{} Radius (Au)'.format(system.upper()), fontsize=fnt)
    ax[1].set_ylabel('{} Lon. (deg)'.format(system.upper()), fontsize=fnt)
    ax[2].set_ylabel('{} Lat. (deg)'.format(system.upper()), fontsize=fnt)

    ax[2].set_xlabel('Date', fontsize=fnt)

    for a in ax:
        a.tick_params("both", labelsize=14)

    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.075, top=0.99, wspace=0.01, hspace=0.0)
    proj_dirs = swp.project_info()
    out_name = "spacecraft_orbits.png"
    out_path = os.path.join(proj_dirs['figs'], out_name)
    fig.savefig(out_path)
    return


def get_hi_files():
    """
    Function used to find a a days worth of HI1a files stored locally for  specific plot.
    :return:
    """
    proj_dirs = swp.project_info()
    hi_path = os.path.join(proj_dirs['data'], "hi1a")
    hi_path = os.path.join(hi_path, '*.fts')
    out_files = glob.glob(hi_path)
    return out_files


def HI_processing_schematic():
    """
    Produces a plot used to illustrate the processing of the HI1a data. To reproduce this exact plot requires
    downloading the HI1a data
    """
    pa_lo = 60
    pa_hi = 120
    dpa = 5.0
    r_lo = 20.0
    r_hi = 22.5

    # Get files for this craft, between time limits
    hi_files = get_hi_files()

    # Split the files into current and previous, for making differenced images
    files_c = hi_files[1:]
    files_p = hi_files[0:-1]

    # Setup position angle bins
    pa_bins = np.arange(pa_lo, pa_hi + dpa, 1.0)
    # Get blank column
    z = np.zeros(len(files_c), dtype=np.float64) * np.NaN
    # Make dict to build the array
    data = {"pa_{:03d}".format(np.int32(pa)): z for pa in pa_bins}
    # Add in time axis
    data['time'] = z

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    pa_cols = swp.get_pa_colors()
    std_all = np.zeros((len(files_c), pa_bins.size)) * np.NaN

    # Loop over files, look at stats in shell on the differenced images and fill
    sel_frame = 28
    for i, (fc, fp) in enumerate(zip(files_c, files_p)):

        # Get the map and the image
        himap = hip.get_image_diff(fc, fp, align=True, smoothing=True)
        # Get arrays of pixel coordinates and convert to HPC
        if i == 0:
            x = np.arange(0, himap.dimensions.x.value)
            y = np.arange(0, himap.dimensions.y.value)
            xm, ym = np.meshgrid(x, y)

        coords = himap.pixel_to_world(xm * u.pix, ym * u.pix)
        # Do my conversion to HPR coords, and then convert to plane of sky distance (r = d*tan(elon))
        el, pa = hip.convert_hpc_to_hpr(coords.Tx, coords.Ty)
        r_pos = ((himap.meta['dsun_obs'] * u.m) * np.tan((el.to('rad').value))) / sun.constants.radius

        # Look up pixels in specified POS distance window.
        id_r = (r_pos.value > r_lo) & (r_pos.value < r_hi)

        wid = 512
        y_lo = wid - wid / 2
        y_hi = wid + wid / 2
        x_lo = wid
        xm2 = xm[y_lo:y_hi, x_lo:]
        ym2 = ym[y_lo:y_hi, x_lo:]
        id_r2 = id_r[y_lo:y_hi, x_lo:]

        if i == sel_frame:
            fig_time_stamp = himap.date.strftime("%Y-%m-%dT%H:%M:%S")
            normalise = mpl.colors.Normalize(vmin=-5e-14, vmax=5e-14)
            img = mpl.cm.gray(normalise(himap.data), bytes=True)
            ax[0, 0].imshow(img, origin='lower')
            roi = mpl.patches.Rectangle((x_lo, y_lo), wid, wid, fill=False, edgecolor='b')
            ax[0, 0].add_patch(roi)
            ax[0, 0].contour(xm2, ym2, id_r2, levels=[0], colors=['r'], linewidths=3, linestyles=['dashed'])

            img = mpl.cm.gray(normalise(himap.data[y_lo:y_hi, x_lo:]), bytes=True)
            ax[0, 1].imshow(img, origin='lower')
            ax[0, 1].contour(xm2 - x_lo, ym2 - y_lo, id_r2, levels=[0], colors=['r'], linewidths=3,
                             linestyles=['dashed'])

        # Preallocate space for the stats in each pa_bin.
        std_arr = np.zeros(pa_bins.shape)
        n_samp_arr = np.zeros(pa_bins.shape)

        for j, pa_b in enumerate(pa_bins):
            # Find this chunk of position angle, and then intersection of the POS and PA windows.
            id_pa = (pa.value > (pa_b - dpa / 2.0)) & (pa.value < (pa_b + dpa / 2.0))
            id_block = id_r & id_pa
            id_block2 = id_block[y_lo:y_hi, x_lo:]

            # Get this sample
            sample = himap.data[id_block].ravel()
            sample = sample * 1e12

            std_arr[j] = np.nanstd(sample)
            n_samp_arr[j] = np.sum(np.isfinite(sample))

            # inspect pas at 75, 90, and 105. Plot out the distributions,
            if i == sel_frame:
                pa_sel = [75, 90, 105]
                style = ['--', '-.', '-']
                pa_plt = {pa_sel[i]: {'col': pa_cols[i], 'style': style[i]} for i in range(3)}

                if pa_b in pa_plt.keys():
                    ax[0, 1].contour(xm2 - x_lo, ym2 - y_lo, id_block2, levels=[0], colors=[pa_plt[pa_b]['col']],
                                     linewidths=3)

                    kde = st.gaussian_kde(sample)
                    diff_I = np.arange(-0.03, 0.03, 0.0005)
                    pdf = kde.pdf(diff_I)
                    std = np.nanstd(sample)
                    avg = np.nanmean(sample)
                    lo = avg - std
                    hi = avg + std
                    pdf_lo = kde.pdf(lo)
                    pdf_hi = kde.pdf(hi)

                    ax[1, 0].plot(diff_I, pdf, color=pa_plt[pa_b]['col'], linestyle=pa_plt[pa_b]['style'],
                                  label="PA = {}".format(pa_b))
                    ax[1, 0].vlines(lo, 0, pdf_lo, linestyle=':', color=pa_plt[pa_b]['col'])
                    ax[1, 0].vlines(hi, 0, pdf_hi, linestyle=':', color=pa_plt[pa_b]['col'])

        std_arr = (std_arr - np.nanmean(std_arr)) / (np.nanstd(std_arr))
        std_all[i, :] = std_arr
        if i == sel_frame:
            ax[1, 1].plot(pa_bins, std_arr, ".-", color='dimgrey', label='Panel B example', zorder=1)
        else:
            ax[1, 1].plot(pa_bins, std_arr, ".-", color='lightgrey', zorder=0)

    std_avg = np.nanmean(std_all, axis=0)
    std_err = 2 * np.nanstd(std_all, axis=0) / np.sqrt(std_all.shape[0])
    ax[1, 1].errorbar(pa_bins, std_avg, yerr=std_err, fmt="ro", ecolor="r", label='Daily mean', zorder=2)

    ax[0, 0].set_xlim(0, 1024)
    ax[0, 0].set_ylim(0, 1024)

    main_fnt = 15
    sub_fnt = 14
    ax[1, 0].set_xlim(-0.028, 0.028)
    ax[1, 0].set_xlabel("Diff. Image Pixel Intensity (Arb. Unit)", fontsize=main_fnt)
    ax[1, 0].set_ylabel("Kernel Density Estimate", fontsize=main_fnt)

    ax[1, 1].set_xlim(pa_lo, pa_hi)
    ax[1, 1].set_xlabel('PA Bin (degrees)', fontsize=main_fnt)
    ax[1, 1].set_ylabel('Diff. Image variability', fontsize=main_fnt)
    ax[1, 1].yaxis.set_label_position("right")
    ax[1, 1].yaxis.tick_right()

    x = 0.015
    y = 0.96
    ax[0, 0].text(x, y, "A) {}".format(fig_time_stamp), transform=ax[0, 0].transAxes, backgroundcolor='k', color='w',
                  fontsize=sub_fnt)
    ax[0, 1].text(x, y, "B)", transform=ax[0, 1].transAxes, backgroundcolor='k', color='w', fontsize=sub_fnt)
    ax[1, 0].text(x, y, "C)", transform=ax[1, 0].transAxes, color='k', fontsize=sub_fnt)
    ax[1, 1].text(x, y, "D)", transform=ax[1, 1].transAxes, color='k', fontsize=sub_fnt)

    ax[1, 0].legend(fontsize=sub_fnt)
    ax[1, 1].legend(fontsize=sub_fnt)

    for a in ax[1, :]:
        a.tick_params("both", labelsize=sub_fnt)

    for a in ax[0, :]:
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.set_xticks([])
        a.set_yticks([])

    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.01, hspace=0.01)
    proj_dirs = swp.project_info()
    out_name = "HI_processing_schematic.png"
    out_path = os.path.join(proj_dirs['figs'], out_name)
    fig.savefig(out_path)
    return


def setup_time_series_acf_figure():
    """
    Setup the figure and axes for the time series and acf plots.
    :return fig, ax, axl, axr: handles to the figure, all axes, left hand axes and right hand axes
    """
    fig = plt.figure(figsize=(15, 10))
    ax1a = plt.subplot2grid((3, 4), (0, 0), colspan=3)
    ax1b = plt.subplot2grid((3, 4), (0, 3), colspan=1)
    ax2a = plt.subplot2grid((3, 4), (1, 0), colspan=3)
    ax2b = plt.subplot2grid((3, 4), (1, 3), colspan=1)
    ax3a = plt.subplot2grid((3, 4), (2, 0), colspan=3)
    ax3b = plt.subplot2grid((3, 4), (2, 3), colspan=1)

    axl = [ax1a, ax2a, ax3a]
    axr = [ax1b, ax2b, ax3b]
    ax = fig.get_axes()
    return fig, ax, axl, axr
