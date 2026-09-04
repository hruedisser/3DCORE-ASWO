import numpy as np

import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LightSource

from .utils import get_iparams_live, extract_row

from ..models.toroidal import ToroidalModel
from ..models.toroidal import thin_torus_gh, thin_torus_qs, thin_torus_sq

import seaborn as sns

sns.set_context("talk")     

sns.set_style("ticks",{'grid.linestyle': '--'})


reference_frame_names = {
    "HEEQ": ['|B|', 'B$_X$', 'B$_Y$', 'B$_Z$'],
    "GSM": ['|B|', 'B$_X$', 'B$_Y$', 'B$_Z$'],
    "RTN": ['|B|', 'B$_R$', 'B$_T$', 'B$_N$'],
    "GSE": ['|B|', 'B$_X$', 'B$_Y$', 'B$_Z$']
}

standard_spacecraft_colors = {
    "psp": "black",
    "solo": "coral",
    "sta": "darkred",
    "stb": "darkgreen",
    "bepi": "blue",
    "wind": "purple",
    "ace": "brown",
    "soho": "grey",
    "noaa_rtsw": "black",
    "noaa_archive": "black"
}

def propagate_model(data_cache, t_snap, t_launch):
    extracted_row = extract_row(data_cache.row)
    iparams = get_iparams_live(*extracted_row)
    model_obj = ToroidalModel(t_launch, **iparams)
    model_obj.generator()
    model_obj.propagator(t_snap)

    return model_obj


def plot_configure(ax, **kwargs):
    view_azim = kwargs.pop("view_azim", -25)
    view_elev = kwargs.pop("view_elev", 25)
    view_radius = kwargs.pop("view_radius", .5)
    
    ax.view_init(azim=view_azim, elev=view_elev)

    ax.set_xlim([-view_radius, view_radius])
    ax.set_ylim([-view_radius, view_radius])
    #adjust scaling as matplotlib lacks automatic aspect ratio setting
    ax.set_zlim([-view_radius*0.75, view_radius*0.75])

    ax.set_axis_off()

    return ax 

def plot_shift(ax,extent,cx,cy,cz):
    #shift center of plot
    ax.set_xbound(cx-extent, cx+extent)
    ax.set_ybound(cy-extent, cy+extent)
    ax.set_zbound(cz-extent*0.75, cz+extent*0.75)

    return ax

def plot_sun(ax, **kwargs):

    kwargs["sunscaler"] = kwargs.pop("sunscaler", 2)
    kwargs["symsize_planet"] = kwargs.pop("symsize_planet", 110)
    kwargs["light_source"] = kwargs.pop("light_source", False)

    # Plot Sun    
    scale = 695510 / 149597870.700 #Rs in km, AU in km
    # sphere with radius Rs in AU
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:30j]
    x = np.cos(u) * np.sin(v) * scale
    y = np.sin(u) * np.sin(v) * scale
    z = np.cos(v) * scale    

    if kwargs["light_source"] == True:
        ls = LightSource(azdeg=320, altdeg=40)  
        ax.plot_surface(x, y, z, rstride=1, cstride=1, color='yellow', lightsource=ls, linewidth=0, antialiased=False, zorder=5)
    else:
        ax.scatter3D(0, 0, 0, color='yellow', s=kwargs["symsize_planet"] *kwargs["sunscaler"], label='Sun',
        edgecolors="black",
        linewidths=0.3,
        zorder=100
        )

    return ax

def plot_planet(ax, data_cache, planet, color, symsize_planet, t_snap):
    
    pos_df = data_cache.body_data[planet]

    if not pos_df.empty:

        time_diffs = np.abs(pos_df['time'] - t_snap)
        closest_index = time_diffs.idxmin()
        x = pos_df.loc[closest_index, 'x']
        y = pos_df.loc[closest_index, 'y']
        z = pos_df.loc[closest_index, 'z']

        ax.scatter3D(x, y, z, color=color, s=symsize_planet, label=planet.capitalize(), zorder=10)

    return ax

def plot_spacecraft(ax, data_cache, spacecraft, color, symsize_spacecraft, t_snap, previous_days = 0, future_days = 0):
    
    pos_df = data_cache.body_data[spacecraft]

    if not pos_df.empty:

        time_diffs = np.abs(pos_df['time'] - t_snap)
        closest_index = time_diffs.idxmin()
        x = pos_df.loc[closest_index, 'x']
        y = pos_df.loc[closest_index, 'y']
        z = pos_df.loc[closest_index, 'z']

        ax.scatter3D(x, y, z, color=color, s=symsize_spacecraft, label=spacecraft.upper(), zorder=10, marker='s')

        if previous_days > 0:
            start_time = t_snap - datetime.timedelta(days=previous_days)
            mask = (pos_df['time'] >= start_time) & (pos_df['time'] <= t_snap)
            ax.plot3D(pos_df.loc[mask, 'x'], pos_df.loc[mask, 'y'], pos_df.loc[mask, 'z'], color=color, lw=1, alpha=0.5, ls='--')
            
        if future_days > 0:
            end_time = t_snap + datetime.timedelta(days=future_days)
            mask = (pos_df['time'] >= t_snap) & (pos_df['time'] <= end_time)
            ax.plot3D(pos_df.loc[mask, 'x'], pos_df.loc[mask, 'y'], pos_df.loc[mask, 'z'], color=color, lw=1, alpha=0.5)

    return ax

def plot_cme(ax, obj, color, lw = 1, alpha = .12, highlight_ecliptic = False, highlight_ecliptic_alpha = 0.3):

    wf_model = obj.visualize_shape(iparam_index = 0)

    wf_array = np.array(wf_model)

    # Extract x, y, z coordinates
    x = wf_array[:, :, 0].flatten()
    y = wf_array[:, :, 1].flatten()
    z = wf_array[:, :, 2].flatten()
    ax.plot_wireframe(*wf_model.T, color=color, linewidth=lw, alpha=alpha)

    if highlight_ecliptic:

        intersecton_wf_model = obj.visualize_shape(iparam_index = 0, resolution = 1)

        intersection_wf_array = np.array(intersecton_wf_model)

        # Highlight the ecliptic plane (z=0)
        x_mesh = intersection_wf_array[:, :, 0]
        y_mesh = intersection_wf_array[:, :, 1]
        z_mesh = intersection_wf_array[:, :, 2]

        ax.contourf(
            x_mesh,
            y_mesh,
            z_mesh,
            levels=[z_mesh.min(), 0],
            zdir="z",
            offset=0,
            colors=[color],
            alpha=highlight_ecliptic_alpha,
        )

    return ax


def visualize_fieldline(obj, q0, index=0, steps=1000, step_size=0.01):
    print('visualize_fieldline')
    """Integrates along the magnetic field lines starting at a point q0 in (q) coordinates and
    returns the field lines in (s) coordinates.

    Parameters
    ----------
    q0 : np.ndarray
        Starting point in (q) coordinates.
    index : int, optional
        Model run index, by default 0.
    steps : int, optional
        Number of integration steps, by default 1000.
    step_size : float, optional
        Integration step size, by default 0.01.

    Returns
    -------
    np.ndarray
        Integrated magnetic field lines in (s) coordinates.
    """

    _tva = np.empty((3,), dtype=obj.dtype)
    _tvb = np.empty((3,), dtype=obj.dtype)

    thin_torus_qs(q0, obj.iparams_arr[index], obj.sparams_arr[index], obj.qs_xs[index], _tva)

    fl = [np.array(_tva, dtype=obj.dtype)]
    def iterate(s):
        thin_torus_sq(s, obj.iparams_arr[index], obj.sparams_arr[index], obj.qs_sx[index],_tva)
        thin_torus_gh(_tva, obj.iparams_arr[index], obj.sparams_arr[index], obj.qs_xs[index], _tvb)
        return _tvb / np.linalg.norm(_tvb)

    while len(fl) < steps:
        # use implicit method and least squares for calculating the next step
        try:
            sol = getattr(least_squares(
                lambda x: x - fl[-1] - step_size *
                iterate((x.astype(obj.dtype) + fl[-1]) / 2),
                fl[-1]), "x")

            fl.append(np.array(sol.astype(obj.dtype)))
        except Exception as e:
            break

    fl = np.array(fl, dtype=obj.dtype)

    return fl


def plot_3dcore_field(ax, obj, step_size=2e-3, q0=[0.8, 0.1, np.pi/2], steps=40000, **kwargs):
    print('Tracing Fieldlines')
    #q0=[0.9, .1, .5]
    #q0i =np.array(q0, dtype=np.float32)
    fl, qfl = obj.visualize_fieldline(q0, index=0,  steps=steps, step_size=step_size, return_phi=True)
    ax.plot(*fl.T, **kwargs)

    diff = qfl[1:-10] - qfl[:-11]
    print("total turns estimates: ", np.sum(diff[diff > 0]) / np.pi / 2)#, np.sum(diff2[diff2 > 0]) / np.pi / 2)

    return ax

def plot_circle(ax, dist, color=None, **kwargs):        

    thetac = np.linspace(0, 2 * np.pi, 100)
    xc = dist * np.sin(thetac)
    yc = dist * np.cos(thetac)
    zc = 0
    ax.plot(xc, yc, zc, color=color, lw=0.3, **kwargs)

def plot_longgrid(ax, fontsize=6, color = 'k', text = True, view_radius = 1):

    radii = [
        #0.3, 
        0.5, 0.8]

    if view_radius < .3:
        multip1 = .225
        multip2 = .25
        radii = [0.2]
    elif view_radius > .9:
        multip1 = 1.2
        multip2 = 1.3
        radii = radii = [0.5, 0.8, 1.]
    else:
        multip1 = .85
        multip2 = .9
    
    for r in radii:
        plot_circle(ax, r, color=color)
        if text == True:
            ax.text(x = -0.085, y = r - 0.15, z = 0, s = f'{r} AU', fontsize = fontsize) 

    # Create data for the AU lines and their labels
    num_lines = 8
    for i in range(num_lines):
        angle_degrees = -180 + (i * 45)  # Adjusted angle in degrees (-180 to 180)
        angle_radians = np.deg2rad(angle_degrees)
        x = [0, np.cos(angle_radians)* multip1] 
        y = [0, np.sin(angle_radians)* multip1]
        z = [0, 0]

        ax.plot(x, y, z, color=color, lw=0.3)

        label_x = multip2 * np.cos(angle_radians)
        label_y = multip2 * np.sin(angle_radians)
        

        if text == True:
            ax.text(x = label_x, y = label_y, z = 0, s = f'+/{angle_degrees}°' if angle_degrees == -180 else f'{angle_degrees}°', fontsize = fontsize, horizontalalignment='center',
     verticalalignment='center') 
            
    return ax

##################################################
##################################################


def plot_insitu_results(
        data_cache = None, 
        reference_frame="HEEQ", 
        delta_time_hours=2,
        insitu_start=None,
        insitu_end=None,
        t_s = None, 
        t_e = None, 
        t_fit = None,
        colors=['#c20078','#f97306', '#069af3'],
        figsize=(20,10),
        fontsize=12,
        lw_insitu=2,
        lw_best=3,
        lw_mean=3,
        lw_fitpts=2,
        ensemble_data = None,
        prediction = False
    ):


    fig, ax = plt.subplots(1, 1, figsize = figsize)
    

    t_data = np.array(data_cache.t_data)

    if insitu_start == None or insitu_end == None:

        start = data_cache.mo_begin - datetime.timedelta(hours=delta_time_hours)
        end = data_cache.endtime + datetime.timedelta(hours=delta_time_hours)

    else:
        start = insitu_start
        end = insitu_end

    time_mask = (t_data >= start) & (t_data <= end)
    t_data = t_data[time_mask]
    
    b_data = data_cache.b_data[reference_frame]
    ref_frame_names = reference_frame_names[reference_frame]

    # Plot ensemble data if given

    c0 = 'k'
    
    if ensemble_data is not None:
        ensemble_data_selected = ensemble_data[reference_frame].copy()
        
        ensemble_data_selected[np.where(ensemble_data_selected == 0)] = np.nan

        ensemble_data_selected = ensemble_data_selected[time_mask]

        perc = 0.95

        b_s2p = np.nanquantile(ensemble_data_selected, 0.5 + perc/2., axis=1)
        b_s2n = np.nanquantile(ensemble_data_selected, 0.5 - perc/2., axis=1)

        b_t = np.sqrt(np.sum(ensemble_data_selected**2, axis=2))

        b_m = np.nanmean(ensemble_data_selected, axis=1)

        b_ts2p = np.nanquantile(b_t, 0.5 + perc/2., axis=1)
        b_ts2n = np.nanquantile(b_t, 0.5 - perc/2., axis=1)

        ax.fill_between(t_data, b_ts2n, b_ts2p, color='k', alpha=0.1)
        ax.fill_between(t_data, b_s2n[:,0], b_s2p[:,0], color=colors[0], alpha=0.25)
        ax.fill_between(t_data, b_s2n[:,1], b_s2p[:,1], color=colors[1], alpha=0.25)
        ax.fill_between(t_data, b_s2n[:,2], b_s2p[:,2], color=colors[2], alpha=0.25)

        bx, by, bz = b_m[:,0], b_m[:,1], b_m[:,2]

        ax.plot(t_data, np.sqrt(bx**2 + by**2 + bz**2), color=c0, lw=lw_best, linestyle="dashed")

        ax.plot(t_data, bx, color=colors[0], alpha=1, lw=lw_best, linestyle="dashed")
        ax.plot(t_data, by, color=colors[1], alpha=1, lw=lw_best, linestyle="dashed")
        ax.plot(t_data, bz, color=colors[2], alpha=1, lw=lw_best, linestyle="dashed")   



    # Plot insitu data

    b_data = b_data[time_mask]


    if prediction:

        # Convert datetime objects to string format once for matching
        t_str = [ti.strftime('%Y-%m-%d-%H-%M') for ti in t_data]
        t_fit_str = t_fit[-1].strftime('%Y-%m-%d-%H-%M')

        # Try to find index of last fitting point
        try:
            tind = t_str.index(t_fit_str)
        except ValueError:
            raise ValueError(f"Fitting timestamp {t_fit_str} not found in t array.")

        # Split time and data arrays at the fitting index
        t_before, b_before = t_data[:tind], b_data[:tind]
        t_after,  b_after  = t_data[tind+1:], b_data[tind+1:]


        # Plot solid lines for data before prediction
        plt.plot(t_before, np.linalg.norm(b_before, axis=1), c0, alpha=0.5, lw=3, label=ref_frame_names[0])
        for i, color in enumerate(colors[:3]):
            plt.plot(t_before, b_before[:, i], color, alpha=1, lw=lw_insitu, label=ref_frame_names[i+1])

        # Plot dotted lines for predicted region
        plt.plot(t_after, np.linalg.norm(b_after, axis=1), c0, ls=':', alpha=0.5, lw=3)
        for i, color in enumerate(colors[:3]):
            plt.plot(t_after, b_after[:, i], color, ls=':', alpha=1, lw=lw_insitu)
    
    else:
        ax.plot(t_data, np.linalg.norm(b_data, axis=1), c0, lw=lw_insitu, label=ref_frame_names[0])
        for i, color in enumerate(colors[:3]):
            plt.plot(t_data, b_data[:, i], color, alpha=1, lw=lw_insitu, label=ref_frame_names[i+1])

    # Plot fitpoints

    if t_s is not None:
        ax.axvline(x=t_s, lw=lw_fitpts, alpha=0.75, color="k", ls="-.")
    if t_e is not None:
        ax.axvline(x=t_e, lw=lw_fitpts, alpha=0.75, color="k", ls="-.")

    if t_fit is not None:
        for t in t_fit:
            ax.axvline(t, color='black', linestyle='--', alpha=0.75, lw=lw_fitpts)



    #ax.set_xlabel('Time', fontsize=fontsize)
    ax.legend(fontsize=fontsize, loc="lower right", ncol=2)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)


    date_form = mdates.DateFormatter("%h %d %H")
    plt.gca().xaxis.set_major_formatter(date_form)
           
    plt.ylabel("B [nT]")
    plt.xticks(rotation=25, ha='right')
    plt.xlim(start,end)

    #ax.grid()

    fig.tight_layout()

    return fig, ax

def plot_3d_results(
        data_caches = [],
        t_snap = None,
        **kwargs
    ):

    kwargs["spacecraft"] = kwargs.pop("spacecraft", {"psp": "black", "solo":"coral", "sta":"darkred", "stb":"darkgreen", "bepi":"blue"})
    kwargs["planets"] = kwargs.pop("planets", {"earth":"mediumseagreen", "mercury":"slategrey", "venus":"darkgoldenrod", "mars":"red"})
    kwargs["symsize_planet"] = kwargs.pop("symsize_planet", 110)
    kwargs["symsize_spacecraft"] = kwargs.pop("symsize_spacecraft", 55)
    kwargs["figsize"] = kwargs.pop("figsize", (10,10))
    kwargs["fontsize"] = kwargs.pop("fontsize", 12)
    kwargs["t_launch"] = kwargs.pop("t_launch", None)
    kwargs["light_source"] = kwargs.pop("light_source", False)
    kwargs["sunscaler"] = kwargs.pop("sunscaler", 2)
    kwargs["add_field"] = kwargs.pop("add_field", True)
    kwargs["view_azim"] = kwargs.pop("view_azim", -25)
    kwargs["view_elev"] = kwargs.pop("view_elev", 25)
    kwargs["view_radius"] = kwargs.pop("view_radius", .5)
    kwargs["view_legend"] = kwargs.pop("view_legend", True)
    kwargs["long_grid"] = kwargs.pop("long_grid", True)

    fig = plt.figure(1, figsize=kwargs["figsize"]) #, layout="constrained")

    ax = fig.add_subplot(111, projection='3d')

    ax = plot_configure(ax, **kwargs)


    # Plot Sun 
    ax = plot_sun(ax, **kwargs)

    for planet, color in kwargs["planets"].items():
        # Plot planet
        ax = plot_planet(ax, data_caches[0], planet, color, kwargs["symsize_planet"], t_snap)

    for sc, color in kwargs["spacecraft"].items():
        # Plot spacecraft
        ax = plot_spacecraft(ax, data_caches[0], sc, color, kwargs["symsize_spacecraft"], t_snap)

    for data_cache in data_caches:

        model_obj = propagate_model(data_cache, t_snap, kwargs["t_launch"])

        print(f"Plotting {data_cache.spacecraft} CME")

        if data_cache.spacecraft in kwargs["spacecraft"].keys():
            color = kwargs["spacecraft"][data_cache.spacecraft]
        else:
            color = standard_spacecraft_colors[data_cache.spacecraft]
        
        ax = plot_cme(ax, model_obj, color)

        if kwargs["add_field"] == True:
            ax = plot_3dcore_field(ax, model_obj, color=color, alpha = .95, lw = .8)

    if kwargs["long_grid"] == True:
        ax = plot_longgrid(ax, fontsize=kwargs["fontsize"], text=True, view_radius = kwargs["view_radius"])
    
    # # --- Auto-adjust axis limits to data ---
    # ax.autoscale(enable=True, tight=True)

    # # Get limits from data
    # x_limits = ax.get_xlim3d()
    # y_limits = ax.get_ylim3d()
    # z_limits = ax.get_zlim3d()

    # # calcualate rations between limits
    # x_range = x_limits[1] - x_limits[0]
    # y_range = y_limits[1] - y_limits[0]
    # z_range = z_limits[1] - z_limits[0]

    # max_range = max(x_range, y_range, z_range)

    # proportional_x_range = x_range / max_range
    # proportional_y_range = y_range / max_range
    # proportional_z_range = z_range / max_range

    # ax.set_box_aspect([proportional_x_range, proportional_y_range, proportional_z_range])

    # ax.set_axis_off()

    if kwargs["view_legend"] == True:
        ax.legend(loc='best', fontsize = kwargs["fontsize"])

    fig.tight_layout()

    return fig, ax


def scatterparams(results_df, figsize=(12,10), fontsize=12):
    
    # drop first column
    results_df.drop(results_df.columns[[0,7, 8,9,10]], axis=1, inplace=True)

    g = sns.pairplot(results_df, 
                     corner=True,
                     plot_kws=dict(marker="+", linewidth=1)
                    )
    g.map_lower(sns.kdeplot, levels=[0.05, 0.32], color=".2") #  levels are 2-sigma and 1-sigma contours

    return g