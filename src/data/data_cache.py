import datetime as datetime
from pathlib import Path

import pandas as pd
import numpy as np

import json
import pickle
import re

import matplotlib.pyplot as plt

from .load_data import (
    load_bepi,
    load_maven,
    load_messenger,
    load_msl,
    load_noaa_archive,
    load_noaa_rtsw,
    load_solo,
    load_stereo_a,
    load_stereo_a_beacon,
    load_stereo_b,
    load_ulysses,
    load_vex,
    load_wind,
)

# === Load data_path from JSON config ===
def load_data_path(config_file=Path(__file__).resolve().parents[2] /'config.json'):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config['data_path']

# Load data_path once globally
data_path = load_data_path()
print(f"Data path loaded: {data_path}")


cache_path = Path(__file__).resolve().parents[2] / "cache"

def get_data_cache(idd = None, mean_hours = 24):
    file_cache_path = cache_path / f"{idd}.p"

    if file_cache_path.exists():
        print(f"Loading data from cache for {idd}")
        data_cache_obj = pickle.load(open(file_cache_path, "rb"))
    else:
        print(f"No cache found for {idd}, loading data from source")
        data_cache_obj = DataCache(idd, mean_hours=mean_hours)
        pickle.dump(data_cache_obj, open(file_cache_path, "wb"))
        print(f"Data cached for {idd} at {file_cache_path}")
    return data_cache_obj

            

class DataCache(object):

    def __init__(self, idd = None, delta_t_days: int = 3, assumed_duration_hours: int = 21, icme_begin=None, mo_begin=None, mo_end=None, mean_hours = 24):
        self.idd = idd

        self.process_dates(idd, delta_t_days, assumed_duration_hours, icme_begin, mo_begin, mo_end)

        if any(name in self.idd for name in ["BEPI", "BepiColombo"]):
            self.spacecraft = "bepi"
            self.b_data, self.pos_data, self.t_data, self.body_data, self.v_data = load_bepi(self.data_begin, self.data_end)
        elif any(name in self.idd for name in ["maven", "MAVEN"]):
            self.spacecraft = "maven"
            self.b_data, self.pos_data, self.t_data, self.body_data, self.v_data = load_maven(self.data_begin, self.data_end)
        elif any(name in self.idd for name in ["messenger", "MESSENGER"]):
            self.spacecraft = "messenger"
            self.b_data, self.pos_data, self.t_data, self.body_data, self.v_data = load_messenger(self.data_begin, self.data_end)
        elif any(name in self.idd for name in ["msl", "MSL"]):
            self.spacecraft = "msl"
            self.b_data, self.pos_data, self.t_data, self.body_data, self.v_data = load_msl(self.data_begin, self.data_end)
        elif any(name in self.idd for name in ["noaa_archive", "NOAA_ARCHIVE"]):
            self.spacecraft = "noaa_archive"
            self.b_data, self.pos_data, self.t_data, self.body_data, self.v_data = load_noaa_archive(self.data_begin, self.data_end)
        elif any(name in self.idd for name in ["noaa_realtime", "NOAA_REALTIME", "NOAA_RTSW"]):
            self.spacecraft = "noaa_rtsw"
            self.b_data, self.pos_data, self.t_data, self.body_data, self.v_data = load_noaa_rtsw(self.data_begin, self.data_end)
        elif any(name in self.idd for name in ["SOLO", "SolO"]):
            self.spacecraft = "solo"
            self.b_data, self.pos_data, self.t_data, self.body_data, self.v_data = load_solo(self.data_begin, self.data_end)
        elif any(name in self.idd for name in ["STEREO_A", "STEREO-A", "STEREO A"]):
            self.spacecraft = "stereo_a"
            self.b_data, self.pos_data, self.t_data, self.body_data, self.v_data = load_stereo_a(self.data_begin, self.data_end)
        elif any(name in self.idd for name in ["STEREO_B", "STEREO-B", "STEREO B"]):
            self.spacecraft = "stereo_b"
            self.b_data, self.pos_data, self.t_data, self.body_data, self.v_data = load_stereo_b(self.data_begin, self.data_end)
        elif any(name in self.idd for name in ["ULYSSES", "Ulysses"]):
            self.spacecraft = "ulysses"
            self.b_data, self.pos_data, self.t_data, self.body_data, self.v_data = load_ulysses(self.data_begin, self.data_end)
        elif any(name in self.idd for name in ["VEX", "Venus Express"]):
            self.spacecraft = "vex"
            self.b_data, self.pos_data, self.t_data, self.body_data, self.v_data = load_vex(self.data_begin, self.data_end)
        elif any(name in self.idd for name in ["WIND", "Wind"]):
            self.spacecraft = "wind"
            self.b_data, self.pos_data, self.t_data, self.body_data, self.v_data = load_wind(self.data_begin, self.data_end)

        if self.v_data is not None:
            t_mask = (np.array(self.t_data) >= self.icme_begin - datetime.timedelta(hours=mean_hours)) & (np.array(self.t_data) < self.icme_begin)
            v_before_event = self.v_data[t_mask]
            self.v_mean_before_event = np.mean(v_before_event)
        else:
            self.v_mean_before_event = None



    def process_dates(self, idd = None, delta_t_days:int=3, assumed_duration_hours: int = 21, icme_begin=None, mo_begin=None, mo_end=None):

        
        if "CUSTOM" in idd:

            # Extract the date using regular expression
            date_pattern = r'(\d{8})'

            match = re.search(date_pattern, idd)
            extracted_date = match.group(1)

            if icme_begin == None:
                icme_begin = datetime.datetime.strptime(extracted_date, '%Y%m%d')

            if mo_begin == None:
                mo_begin = datetime.datetime.strptime(extracted_date, '%Y%m%d')

            if mo_end == None:
                mo_end = mo_begin + datetime.timedelta(hours=assumed_duration_hours)
        
        else:
            url='https://helioforecast.space/static/sync/icmecat/HELIO4CAST_ICMECAT_v23.csv'
            icmecat=pd.read_csv(url)
            idds = icmecat.loc[:,'icmecat_id']
            endtime = icmecat.loc[:,'mo_end_time']
            mobegintime = icmecat.loc[:, 'mo_start_time']
            icmebegintime = icmecat.loc[:, 'icme_start_time']

            dateFormat="%Y-%m-%dT%H:%MZ"
            mobegin = pd.to_datetime(mobegintime, format=dateFormat)
            icmebegin = pd.to_datetime(icmebegintime, format=dateFormat)
            end = pd.to_datetime(endtime, format=dateFormat)
            
            i = np.where(idds == idd)[0]

            icme_begin = icmebegin.iloc[i[0]]
            mo_begin = mobegin.iloc[i[0]]
            mo_end = end.iloc[i[0]]
        
        self.icme_begin = icme_begin
        self.mo_begin = mo_begin
        self.endtime = mo_end

        self.data_begin = self.icme_begin - datetime.timedelta(days=delta_t_days)
        self.data_end = self.endtime + datetime.timedelta(days=delta_t_days)

    def quick_insitu_plot(self, reference_frame='HEEQ', delta_time_hours=2, colors = ["r", "g", "b"]):

        t_data = np.array(self.t_data)
        time_mask = (t_data >= (self.mo_begin - datetime.timedelta(hours=delta_time_hours))) & (t_data <= (self.endtime + datetime.timedelta(hours=delta_time_hours)))
        t_data = t_data[time_mask]
        
        if reference_frame == "HEEQ":
            b_data = self.b_data["HEEQ"]
        elif reference_frame == "GSM":
            b_data = self.b_data["GSM"]
        elif reference_frame == "RTN":
            b_data = self.b_data["RTN"]

        b_data = b_data[time_mask]

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        axes[0].plot(t_data, b_data[:,0], label='Bx', color=colors[0])
        axes[0].plot(t_data, b_data[:,1], label='By', color=colors[1])
        axes[0].plot(t_data, b_data[:,2], label='Bz', color=colors[2])
        axes[0].plot(t_data, np.linalg.norm(b_data, axis=1), label='Btot', color='k')

        axes[0].set_ylabel('Magnetic Field (nT)')
        axes[0].legend()

        v_data = self.v_data[time_mask]
        axes[1].plot(t_data, v_data, label='V', color='k')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Velocity (km/s)')

        return fig, axes
    
    def check_fitting_points(self, t_s, t_e, t_fit, reference_frame='HEEQ', delta_time_hours=2, colors = ["r", "g", "b"]):

        if t_s == None:
            t_s = self.mo_begin
        if t_e == None:
            t_e = self.endtime

        fig, axes = self.quick_insitu_plot(reference_frame=reference_frame, delta_time_hours=delta_time_hours, colors = colors)

        axes[0].axvspan(t_s, t_e, color='grey', alpha=0.3)

        axes[0].axvline(t_s, color='red', linestyle='--')
        axes[0].axvline(t_e, color='red', linestyle='--')

        for t in t_fit:
            axes[0].axvline(t, color='black', linestyle='--')

        return fig, axes
    
    def quick_positions_plot(self, spacecraft = {"psp": "black", "solo":"coral", "sta":"darkred", "stb":"darkgreen", "bepi":"blue"}, planets = {"earth":"mediumseagreen", "mercury":"slategrey", "venus":"darkgoldenrod", "mars":"red"}, symsize_planet=110, symsize_spacecraft=55):

        fig = plt.figure(1, figsize=(14,10))

        ax = fig.add_subplot(projection="polar")

        # Plot Sun 
        ax.scatter(
            0,
            0,
            s=180,
            label="Sun",
            alpha=1,
            color="yellow",
            edgecolors="k",
            linewidths=0.3
        )

        # Plot planets
        for planet, color in planets.items():
            pos_df = self.body_data[planet]

            # check if pos_df is not empty
            if not pos_df.empty:
                #find closest time to the event begin
                time_diffs = np.abs(pos_df['time'] - self.mo_begin)
                closest_index = time_diffs.idxmin()
                r = pos_df.loc[closest_index, 'r']
                # lon is in radians
                lon =pos_df.loc[closest_index, 'lon']
                # get lat corrections
                lat = pos_df.loc[closest_index, 'lat']
                r_corr = r * np.cos(lat)

                ax.scatter(
                    lon,
                    r_corr,
                    s=symsize_planet,
                    label=planet.capitalize(),
                    alpha=1,
                    color=color,
                    marker="o",
                )
        
        for sc, color in spacecraft.items():
            pos_df = self.body_data[sc]

            # check if pos_df is not empty
            if not pos_df.empty:

                #find closest time to the event begin
                time_diffs = np.abs(pos_df['time'] - self.mo_begin)
                closest_index = time_diffs.idxmin()
                r = pos_df.loc[closest_index, 'r']
                # lon is in radians
                lon = pos_df.loc[closest_index, 'lon']
                # get lat corrections
                lat = pos_df.loc[closest_index, 'lat']
                r_corr = r * np.abs(np.cos(lat))

                ax.scatter(
                    lon,
                    r_corr,
                    s=symsize_spacecraft,
                    label=sc.upper(),
                    alpha=1,
                    color=color,
                    marker="s"
                )

                ax.plot(pos_df.loc[:closest_index, "lon"], pos_df.loc[:closest_index, "r"], color=color, linestyle='-', alpha=0.7)
                ax.plot(pos_df.loc[closest_index:, "lon"], pos_df.loc[closest_index:, "r"], color=color, linestyle='--', alpha=0.7)
        


        ax.set_theta_zero_location("E")
        ax.set_ylim(0, 1.2)
        ax.legend()

        return fig, ax
    