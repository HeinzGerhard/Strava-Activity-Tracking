
import matplotlib.pyplot as plt
import gpxpy
import gpxpy.gpx
import glob
import numpy as np
from fit_tool.fit_file import FitFile
from fit_tool.profile.messages.record_message import RecordMessage
from fit_tool.profile.messages.sport_message import SportMessage
from fit_tool.profile.messages.activity_message import ActivityMessage
from datetime import datetime
from datetime import timedelta
import geopy.distance
import os
import operator
import plotly.express as px
import pandas as pd
import pickle
import pathlib
import time as timeit
import rasterio
import utm
import gzip
import shutil
import Tiles
import Calculate_Polygons


buffered_nasa_sets = {} # Buffer for the 30m NASA DEM Models https://dwtkns.com/srtm30m/
debug = False
# Load the DEM Model for Norway
datasets = []
files = glob.glob("./DEM/**/*.tif", recursive=True)# 50 M Model for the rest of Norway
for file in files:
    datasets.append(rasterio.open(file))

class Activity:
    def __init__(self):
        self.line = []
        self.polygons = []
        self.intersections = []
        self.times = []
        self.temperature = []
        self.speed = []
        self.elevation = []
        self.distance = []
        self.duration = []
        self.lat = []
        self.lon = []
        self.cadence = []
        self.accumulated_power = []
        self.left_pedal_smoothness = []
        self.left_torque_effectiveness = []
        self.heart_rate = []
        self.heart_rate_zone = []
        self.power = []
        self.power_zone = []
        self.datetime = []
        self.dt = []
        self.ds = []
        self.time_since_start = []
        self.year = 0
        self.virtual = False
        self.timestamp = 0
        self.name = 'FIT'
        self.hr_zones = [0,0,0,0,0]
        self.power_zones = [0,0,0,0,0,0,0]
        self.power_curve = [[0], [0]]
        self.temperatue_zones = np.zeros(61)
        self.sport = 'sport'
        self.name = ''
        self.date = 0
        self.file = ''
        self.gear = ''
        self.commute = False

    def read_tcx(self,file):

        self.file = file
        file = pathlib.Path(file)
        from xml.dom import minidom
        xmldoc = minidom.parse('./data/activities/12490115472.tcx')
        tcd = xmldoc.getElementsByTagName("TrainingCenterDatabase")[0]
        activitiesElement = tcd.getElementsByTagName("Activities")[0]
        activities = activitiesElement.getElementsByTagName("Activity")
        activity = activities[0]
        sport = activity.attributes["Sport"]
        sportName = sport.value
        idElement = activity.getElementsByTagName("Id")[0]
        timeOfDay = idElement.firstChild.data
        year = int(timeOfDay[0:4])
        month = int(timeOfDay[5:7])
        day = int(timeOfDay[8:10])
        date = datetime(year, month, day)
        #date.tz_localize(None)
        # print(sportName, month, day, year)
        print(sportName, date)

        self.sport = sportName
        self.virtual = False

        self.year = year
        self.timestamp = date.timestamp()

        trackpoints = activity.getElementsByTagName('Trackpoint')
        for trackpoint in trackpoints:
            position = trackpoint.getElementsByTagName('Position')[0]
            lat = float(position.getElementsByTagName('LatitudeDegrees')[0].firstChild.data)
            long = float(position.getElementsByTagName('LongitudeDegrees')[0].firstChild.data)

            altitude = float(trackpoint.getElementsByTagName('AltitudeMeters')[0]._get_firstChild().data)
            time= pd.to_datetime(trackpoint.getElementsByTagName('Time')[0]._get_firstChild().data).tz_localize(None)
            distance= float(trackpoint.getElementsByTagName('DistanceMeters')[0]._get_firstChild().data)
            speed = float(trackpoint.getElementsByTagName('Extensions')[0].getElementsByTagName('ns3:TPX')[0].getElementsByTagName('ns3:Speed')[0]._get_firstChild().data)
            hr =float(trackpoint.getElementsByTagName('HeartRateBpm')[0].getElementsByTagName('Value')[0]._get_firstChild().data)
            self.times.append(time.timestamp())
            self.datetime.append(time)
            self.temperature.append(np.nan)
            self.distance.append((distance) / 1000)
            self.elevation.append(altitude)
            self.speed.append(speed)
            self.lat.append(lat)
            self.lon.append(long)
            self.heart_rate.append(hr)
            self.power.append(np.nan)
            self.cadence.append(np.nan)
            self.accumulated_power.append(np.nan)
            self.left_pedal_smoothness.append(np.nan)
            self.left_torque_effectiveness.append(np.nan)
        self.normalize_activity(file)

    def read_fit(self, file):
        start = timeit.time()
        self.file = file
        file = pathlib.Path(file)
        print(f'\t\tLoading Fit activity file {file}')
        app_fit = FitFile.from_file(file)

        loaded = timeit.time()
        old_dist = 0
        for record in app_fit.records:
            message = record.message
            if isinstance(message, SportMessage):
                if message.sport_name == 'INDOOR':
                    self.virtual = True
                    self.sport = 'virtual cycling'
                elif message.sport_name == 'ROAD' or message.sport_name == 'TOUR':
                    self.virtual = False
                    self.sport = 'cycling'
                else:
                    self.virtual = False
                    self.sport = message.sport_name
                self.sport = message.sport_name

            if isinstance(message, ActivityMessage):
                time = datetime.fromtimestamp(int(message.timestamp) / 1000)
                self.year = time.year
                self.timestamp = message.timestamp / 1000
            if isinstance(message, RecordMessage):
                if message.speed is None:
                    if message.distance is not None:
                        old_dist = (message.distance or 0) if message.distance < 10000000 else None
                else:
                    self.times.append(message.timestamp / 1000)
                    self.datetime.append(pd.to_datetime(message.timestamp,unit='ms'))
                    self.temperature.append(message.temperature)
                    self.distance.append((message.distance or old_dist)/1000)
                    self.elevation.append(fix_elevation_point(message.position_lat, message.position_long, message.altitude))
                    self.speed.append(message.speed if (message.speed or 0) < 200 else 0)
                    self.lat.append(message.position_lat)
                    self.lon.append(message.position_long)
                    self.heart_rate.append(message.heart_rate)
                    self.power.append(message.power if message.power == None else message.power if message.power < 2500 else 0)
                    self.cadence.append(message.cadence)
                    self.accumulated_power.append(message.accumulated_power)
                    self.left_pedal_smoothness.append(message.left_pedal_smoothness)
                    self.left_torque_effectiveness.append(message.left_torque_effectiveness)

        self.normalize_activity(file)
        end = timeit.time()
        print(f'Duration File load: {end - start}, parsing: {loaded - start}' )

    def load_file(self, file):
        self.file=file
        file = pathlib.Path(file)
        print(f'\t\tLoading GPX activity file {file}')
        with open(file, 'r') as gpx_file:
            gpx = gpxpy.parse(gpx_file)

            if gpx.tracks.__len__() >0:
                if 'Zwift' in gpx.tracks[0].name:
                    self.virtual = True
                    print('\t\t\tVirtual Ride')
                else:
                    self.virtual = False
                if not gpx.time is None:
                    self.year = gpx.time.year
                else:
                    self.year = 2021
                for track in gpx.tracks:
                    self.sport = track.type
                    self.name = track.name
                    for segment in track.segments:
                        for point in segment.points:
                            # print(f'Point at ({point.latitude},{point.longitude}) -> {point.elevation}')
                            self.elevation.append(point.elevation)
                            self.speed.append(point.speed)
                            self.lat.append(point.latitude)
                            self.lon.append(point.longitude)
                            self.temperature.append(None)
                            self.power.append(None)
                            self.heart_rate_zone.append(None)
                            self.power_zone.append(None)
                            self.cadence.append(np.nan)
                            self.accumulated_power.append(np.nan)
                            self.left_pedal_smoothness.append(np.nan)
                            self.left_torque_effectiveness.append(np.nan)
                            if point.time is not None:
                                gpx_timestamp = point.time.timestamp()
                            else:
                                gpx_timestamp = datetime.now().timestamp()
                            self.times.append(gpx_timestamp)
                            self.datetime.append(pd.to_datetime(gpx_timestamp,unit='s'))
                            hr = None
                            for el in point.extensions:
                                if el.__len__() > 0:
                                    el = el[0]
                                    if 'hr' in el.tag:
                                        hr = int(el.text)
                            self.heart_rate.append(hr)

                self.speed = np.array(self.speed, dtype=np.float16)
                self.lat = np.array(self.lat, dtype=np.float32)
                self.lon = np.array(self.lon, dtype=np.float32)
                self.times = np.array(self.times)
                self.distance = [0]
                if not max(self.speed) > 0:
                    self.speed = [0]
                    for idx, lat in enumerate(self.lat[1:]):
                        distance = geopy.distance.geodesic([self.lat[idx], self.lon[idx]],
                                                                     [self.lat[idx + 1], self.lon[idx + 1]]).km
                        self.distance.append(distance + max(self.distance))
                        speed = distance/(self.times[idx +1]-self.times[idx])*1000
                        if speed > 60:
                            speed = np.nan
                        self.speed.append(speed)
                    self.speed = np.array(self.speed, dtype=np.float16)
                else:
                    for idx, lat in enumerate(self.lat[1:]):
                        self.distance.append(geopy.distance.geodesic([self.lat[idx], self.lon[idx]],
                                                                     [self.lat[idx + 1], self.lon[idx + 1]]).km + max(
                            self.distance))

                self.normalize_activity(file)
            else:
                print('No Track for Activity')

    def normalize_activity(self, file):

        self.times = np.array(self.times, dtype=float)
        try:
            self.time_since_start = (self.times-min(self.times))/60
        except:
            pass
        self.elevation = np.array(self.elevation, dtype=np.float16)
        self.speed = np.array(self.speed, dtype=np.float16)
        self.lat = np.array(self.lat, dtype=np.float32)
        self.lon = np.array(self.lon, dtype=np.float32)
        self.temperature = np.array(self.temperature, dtype=np.float16)
        self.distance = np.array(self.distance, dtype=np.float16)
        self.power = np.array(self.power, dtype=np.float16)
        self.heart_rate = np.array(self.heart_rate, dtype=np.float16)
        self.datetime = np.array(self.datetime)
        self.cadence = np.array(self.cadence, dtype=np.float16)
        self.accumulated_power = np.array(self.accumulated_power, dtype=np.float32)
        self.left_pedal_smoothness = np.array(self.left_pedal_smoothness, dtype=np.float16)
        self.left_torque_effectiveness = np.array(self.left_torque_effectiveness, dtype=np.float16)

        time = datetime.fromtimestamp(min(self.times.astype(float)))  
        self.year = time.year

        if file.stem.replace('_',' ').isnumeric():
            self.name = time.strftime(f"%Y-%m-%d, %H:%M {self.sport}")
        else:
            self.name = time.strftime("%Y-%m-%d, %H:%M ") + file.stem.replace('_',' ')
        self.date = int(time.strftime("%Y%m%d"))

        self.calculate_dt()
        self.get_HR_zones()
        self.get_power_zones()
        self.get_power_curve()

    def calculate_dt(self):
        self.dt = [0]
        self.ds = [0]
        for idx, time in enumerate(self.times[1:]):
            dt =  time - self.times[idx]
            if dt > 20:
                dt = 0
                ds = 0
            else:
                ds = self.distance[idx + 1] - self.distance[idx]
            self.ds.append(ds)
            self.dt.append(dt)
        self.dt = np.array(self.dt, dtype=np.float32)
        self.ds = np.array(self.ds, dtype=np.float32)

    def get_HR_zones(self):
        self.heart_rate_zone = []
        max_hr = 185
        zones = [0,0,0,0,0]
        for idx, hr in enumerate(self.heart_rate):
            zone = 0
            if hr is not None:
                time = self.dt[idx]
                if hr < 0.60*max_hr:
                    zone = 1
                    zones[0] = zones[0]+time
                elif hr < 0.70*max_hr:
                    zone = 2
                    zones[1] = zones[1]+time
                elif hr < 0.80*max_hr:
                    zone = 3
                    zones[2] = zones[2]+time
                elif hr < 0.90*max_hr:
                    zone = 4
                    zones[3] = zones[3]+time
                elif hr < 255:
                    zone = 5
                    zones[4] = zones[4]+time
            self.heart_rate_zone.append(zone)
        self.hr_zones = zones
        return zones

    def get_power_zones(self):
        self.power_zone = []
        ftp = 250
        zones = [0,0,0,0,0,0,0]
        for idx, power in enumerate(self.power):
            if power is not None:
                time = self.dt[idx]
                if power < 0.55*ftp:
                    zone = 1
                    zones[0] = zones[0]+time
                elif power < 0.75*ftp:
                    zone = 2
                    zones[1] = zones[1]+time
                elif power < 0.87*ftp:
                    zone = 3
                    zones[2] = zones[2]+time
                elif power < 0.94*ftp:
                    zone = 4
                    zones[3] = zones[3]+time
                elif power < 1.05*ftp:
                    zone = 5
                    zones[4] = zones[4]+time
                elif power < 1.20*ftp:
                    zone = 6
                    zones[5] = zones[5]+time
                else:
                    zone = 7
                    zones[6] = zones[6]+time
                self.power_zone.append(zone)
        self.power_zones = zones
        return zones


    def get_power_curve(self):
        durations = [1, 5, 10, 30, 60, 120, 180, 300, 600, 900, 1200, 1800, 2400, 3600, 4800,7200,3600*3,3600*4,3600*10]
        results = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        start_time = min(self.times)
        end_time = max(self.times)
        activity_duration = end_time - start_time
        if not np.isnan(np.nanmax(self.power)):
            for idx, duration in enumerate(durations):
                if duration < activity_duration:
                    time = start_time
                    max_power = 0
                    while time < (end_time - duration):
                        power = self.power[(self.times >= time) & (self.times < time + duration)]
                        max_power = max(max_power, power.mean())
                        time = time + 1
                    results[idx] = np.nanmax([max_power,0])
            self.power_curve = [results, durations]
        else:
            print('\t\tNo Power measurements')
        return [results, durations]

    def fix_duration(self, duration):
        fix = np.zeros(self.time_since_start.size)
        for idx, time in enumerate(self.time_since_start[1:]):
            gap = time - self.time_since_start[idx]
            addition = 0
            if gap > duration:
                print(f'\t\t\tRemoved Stop of {round(gap/60,2)} hours')
                addition = gap
            fix[idx+1] = fix[idx] - addition
        self.time_since_start = np.add(self.time_since_start,fix)

    def crop_activity(self, date, duration):
        try:
            end = date + pd.Timedelta(seconds=(duration + 1))
            if min(self.datetime) < date or max(self.datetime) > end:
                if debug:
                    print(f'Potential Crop {self.name}')
                mask = np.logical_and(self.datetime >= date, self.datetime <= end)
                if mask.__contains__(False):
                    if debug:
                        plt.close('All')
                        print(f'To be cropped {mask = }')
                        plt.plot(self.lon, self.lat)
                        plt.plot(self.lon[mask], self.lat[mask])
                        plt.title(f'{self.name}, cropping')
                        plt.show()
                    self.times = self.times[mask]
                    self.temperature = self.temperature[mask]
                    self.speed = self.speed[mask]
                    self.elevation = self.elevation[mask]
                    self.distance = self.distance[mask]
                    self.lat = self.lat[mask]
                    self.lon = self.lon[mask]
                    self.heart_rate = self.heart_rate[mask]
                    self.heart_rate_zone = np.array(self.heart_rate_zone)[mask]
                    self.power = self.power[mask]
                    self.power_zone = np.array(self.power_zone)[mask]
                    self.datetime = self.datetime[mask]
                    self.dt = self.dt[mask]
                    self.ds = self.ds[mask]
                    self.time_since_start = self.time_since_start[mask]
                    self.cadence = self.cadence[mask]
                    self.accumulated_power = self.accumulated_power[mask]
                    self.left_pedal_smoothness = self.left_pedal_smoothness[mask]
                    self.left_torque_effectiveness = self.left_torque_effectiveness[mask]
        except:
            print(f'Did not crop {self.name}')

    def reduce(self):
        self.times = []
        self.temperature = []
        self.speed = []
        self.elevation = []
        self.distance = []
        self.duration = []
        self.lat = []
        self.lon = []
        self.heart_rate = []
        self.heart_rate_zone = []
        self.power = []
        self.power_zone = []
        self.datetime = []
        self.dt = []
        self.time_since_start = []


def plot_data(activities):
    d = {'Lat': [], 'Long': [], 'alt': [], 'speed': [], 'year': [], 'time': [], 'distance': [], 'total_distance': []}
    all_data = pd.DataFrame(data=d)
    tot_distance = 0
    for activity in activities:
        if activity.year == 2023 and not activity.virtual:
            d = {'Lat': activity.lat,
                 'Long': activity.lon,
                 'alt': activity.elevation,
                 'speed': activity.speed,
                 'time': activity.times,
                 'year': activity.year,
                 'distance': activity.distance,
                 'total_distance': np.array(activity.distance) + tot_distance}
            new_data = pd.DataFrame(data=d)
            tot_distance = max(new_data['total_distance'])
            all_data = pd.concat([all_data, new_data])

    # plt.plot([datetime.fromtimestamp(ts) for ts in all_data['time']], all_data['total_distance'])
    fig = px.scatter_mapbox(all_data.iloc[::5, :],
                            lat="Lat",
                            lon="Long",
                            # hover_name="Address",
                            # hover_data=["Address", "Listed"],
                            # color="alt",
                            # color="distance",
                            color="time",
                            color_continuous_scale='viridis',
                            # size="Listed",
                            zoom=2,
                            height=1200,
                            width=1200)

    # fig.update_layout(mapbox_style="open-street-map")
    # fig.update_layout(mapbox_style="carto-positron")
    fig.update_layout(mapbox_style="carto-darkmatter")
    # fig.update_layout(mapbox_style="stamen-terrain")
    # fig.update_layout(mapbox_style="stamen-toner")
    # fig.update_layout(mapbox_style="stamen-watercolor")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()

    d = {'Lat': [], 'Long': [], 'alt': [], 'speed': [], 'year': [], 'time': [], 'distance': [], 'total_distance': [], 'date': [], 'day': []}
    all_data = pd.DataFrame(data=d)
    tot_distance = 0
    for idx, activity in enumerate(activities):
        if not activity.virtual:
            activity.lat = np.append(activity.lat, [None, None, None, None, None])
            activity.lon = np.append(activity.lon, [None, None, None, None, None])
            activity.elevation = np.append(activity.elevation, [None, None, None, None, None])
            activity.speed = np.append(activity.speed, [None, None, None, None, None])
            activity.times = np.append(activity.times, [activity.times[-1], activity.times[-1], activity.times[-1], activity.times[-1], activity.times[-1]])
            activity.distance = np.append(activity.distance, [max(activity.distance), max(activity.distance), max(activity.distance), max(activity.distance), max(activity.distance)])
            date = []
            for ts in activity.times:
                if ts is not None:
                    date.append(datetime.fromtimestamp(ts))
                else:
                    date.append(None)
            day = []
            for ts in date:
                if ts is not None:
                    day.append(ts.day)
                else:
                    day.append(None)
            d_new = {'Lat': activity.lat,
                 'Long': activity.lon,
                 'alt': [int(el) for el in np.nan_to_num(activity.elevation)],
                 'speed': activity.speed,
                 'time': activity.times,
                 'date': date,
                 'day': day,
                 'year': activity.year,
                 'distance': activity.distance,
                 'total_distance': np.array(activity.distance) + tot_distance}
            new_data = pd.DataFrame(data=d_new)
            tot_distance = max(new_data['total_distance'])
            all_data = pd.concat([all_data, new_data.iloc[::5, :]])

    fig = px.scatter_mapbox(all_data.iloc[::5, :],
                            lat="Lat",
                            lon="Long",
                            color="year",
                            color_continuous_scale='viridis',
                            # size="Listed",
                            zoom=2,
                            height=1200,
                            width=1200)

    # fig.update_layout(mapbox_style="open-street-map")
    # fig.update_layout(mapbox_style="carto-positron")
    fig.update_layout(mapbox_style="carto-darkmatter")
    # fig.update_layout(mapbox_style="stamen-terrain")
    # fig.update_layout(mapbox_style="stamen-toner")
    # fig.update_layout(mapbox_style="stamen-watercolor")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()

    fig = px.scatter_mapbox(all_data.iloc[::5, :],
                            lat="Lat",
                            lon="Long",
                            color="day",
                            #color_continuous_scale='viridis',
                            zoom=2,
                            height=1200,
                            width=1200)
    fig.update_layout(mapbox_style="carto-positron")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()

    fig = px.line_mapbox(all_data,
                            lat="Lat",
                            lon="Long",
                            color="day",
                            zoom=8,
                            height=1200,
                            width=1200)

    fig.update_layout(mapbox_style="carto-positron")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()

    fig = px.density_mapbox(all_data.iloc[::1, :], lat='Lat', lon='Long', radius=2,
                            center=dict(lat=0, lon=180), zoom=0,
                            mapbox_style="carto-darkmatter")
    fig.show()


def plot_hr(activities):

    zones = np.array([0,0,0,0,0])

    d = {'Lat': [],
         'Long': [],
         'Heart Rate': []}
    all_df = pd.DataFrame(data=d)

    for activity in activities:

        zones = zones + np.array(activity.hr_zones)

        d = {'Lat': activity.lat,
             'Long': activity.lon,
             'Heart Rate': activity.heart_rate}
        df = pd.DataFrame(data=d)
        all_df = pd.concat([all_df, df])

    plt.bar(['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4', 'Zone 5'], zones)
    plt.show()

    fig = px.scatter_mapbox(all_df,
                            lat="Lat",
                            lon="Long",
                            color="Heart Rate",
                            color_continuous_scale='rainbow',
                            zoom=5,
                            #showscale=True,
                            range_color =[50,min(170, max(all_df['Heart Rate']))],
                            height=1200,
                            width=1200)
    fig.update_layout(mapbox_style="carto-positron")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()


def plot_power(activities):

    zones = np.array([0,0,0,0,0,0,0])
    curve =  np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    d = {'Lat': [],
         'Long': [],
         'Power': []}
    all_df = pd.DataFrame(data=d)

    for activity in activities:
        if activity.virtual:
            print('Virtual')
            plt.plot(activity.power_curve[1], activity.power_curve[0])
            zones = zones + np.array(activity.power_zones)
            curve = np.maximum(curve,activity.power_curve[0])
            d = {'Lat': activity.lat,
                 'Long': activity.lon,
                 'Power': activity.power}
            df = pd.DataFrame(data=d)
            all_df = pd.concat([all_df, df])

    plt.plot(activity.power_curve[1], curve)
    plt.show()
    plt.plot(activity.power_curve[1], curve)
    plt.xscale('log')
    plt.grid()
    plt.show()

    plt.bar(['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4', 'Zone 5', 'Zone 6', 'Zone 7'], zones)
    plt.show()

    fig = px.scatter_mapbox(all_df,
                            lat="Lat",
                            lon="Long",
                            color="Power",
                            color_continuous_scale='rainbow',
                            zoom=5,
                            size="Power",
                            range_color =[0,min(350, max(all_df['Power']))],
                            height=1200,
                            width=1200)
    fig.update_layout(mapbox_style="carto-positron")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()


def plot_activity_type(activities):
    d = {'Lat': [],
         'Long': [],
         'Type': []}
    all_df = pd.DataFrame(data=d)

    for activity in activities:

            d = {'Lat': activity.lat,
                 'Long': activity.lon,
                 'Type': activity.sport}
            df = pd.DataFrame(data=d)
            all_df = pd.concat([all_df, df])

    fig = px.scatter_mapbox(all_df.iloc[::5, :],
                            lat="Lat",
                            lon="Long",
                            color="Type",
                            zoom=5,
                            height=1200,
                            width=1200)
    fig.update_layout(mapbox_style="carto-positron")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()


def plot_activity_times(activities):
    d = {'Lat': [],
         'Long': [],
         'Time': []}
    all_df = pd.DataFrame(data=d)

    for activity in activities:

            d = {'Lat': activity.lat,
                 'Long': activity.lon,
                 'Time': activity.times}
            df = pd.DataFrame(data=d)
            all_df = pd.concat([all_df, df])

    fig = px.scatter_mapbox(all_df.iloc[::5, :],
                            lat="Lat",
                            lon="Long",
                            color="Time",
                            zoom=5,
                            height=1200,
                            width=1200)
    fig.update_layout(mapbox_style="carto-positron")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()


def plot_activity_elevation(activities):
    d = {'Lat': [],
         'Long': [],
         'Time': [],
         'Elevation': [],
         'Speed': []}
    all_df = pd.DataFrame(data=d)

    for activity in activities:

            d = {'Lat': activity.lat,
                 'Long': activity.lon,
                 'Time': activity.times,
                 'Elevation': activity.elevation,
                 'Speed': activity.speed}
            df = pd.DataFrame(data=d)
            all_df = pd.concat([all_df, df])

    fig = px.scatter_mapbox(all_df.iloc[::5, :],
                            lat="Lat",
                            lon="Long",
                            color="Elevation",
                            color_continuous_scale='rainbow',

                            range_color=[0, min(1000, max(all_df['Elevation']))],
                            #size='Speed',
                            zoom=5,
                            height=1200,
                            width=1200)
    fig.update_layout(mapbox_style="carto-positron")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()


def extract_activities(df, activity_name, append):
    dff = df[df['Name'] == (activity_name[0])]
    for idx, name in enumerate(activity_name):
        if idx > 0:
            dfn = df[df['Name'] == name]
            if append == 'Append':
                dfn['Duration'] = (dfn['Time'] - min(dff['Time'])) / 60
                dfn['Distance'] = dfn['Distance'] + max(dff['Distance'])
            elif append == 'Remove Gap':
                dfn['Duration'] = dfn['Duration'] + max(dff['Duration'])
                dfn['Distance'] = dfn['Distance'] + max(dff['Distance'])
            dff = pd.concat([dff, dfn])
    return dff

def get_subdataset(df, start_date, end_date, display, activity_name, append, virtual, activity_type):
    if start_date.__contains__('T'): start_date = start_date.split('T')[0]
    if end_date.__contains__('T'): end_date = end_date.split('T')[0]
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date) + timedelta(days=1)
    end.replace(hour=0, microsecond=0, second=0)

    dff = df[df['DateTime'] >= start]
    dff = dff[dff['DateTime'] <= end]

    if display != 'All' and activity_name.__len__() > 0:
        dff = extract_activities(dff, activity_name, append)

    if virtual == 'Outdoors':
        dff = dff[dff['Virtual'] == False]
    elif virtual == 'Virtual':
        dff = dff[dff['Virtual'] == True]

    dff = dff[dff['Type'].isin(activity_type)]
    return dff

def plot_temperature(activities):

    labels = np.array(range(-20,41))
    zones =  np.zeros(labels.size)
    d = {'Lat': [],
         'Long': [],
         'Temperature': []}
    all_df = pd.DataFrame(data=d)

    for activity in activities:
        if not activity.virtual and max(activity.temperatue_zones) > 0:

            zones = zones + np.array(activity.temperatue_zones)
            d = {'Lat': activity.lat,
                 'Long': activity.lon,
                 'Temperature': activity.temperature}
            df = pd.DataFrame(data=d)
            all_df = pd.concat([all_df, df])

    plt.bar(labels[9:55], zones[9:55])
    plt.show()
    labels_combined = [-10, 0, 10, 20, 30]
    zones_combined = [np.sum(zones[10:20]), np.sum(zones[20:30]), np.sum(zones[30:40]), np.sum(zones[40:50]),
                      np.sum(zones[50:60])]
    plt.bar(labels_combined, zones_combined)
    plt.grid()
    plt.show()
    zones_comb_ratio = zones_combined / np.sum(zones_combined)
    fig, ax = plt.subplots()
    ax.bar(labels_combined, zones_comb_ratio)
    ax.set_yscale('log')
    plt.grid()
    plt.show()
    fig = px.scatter_mapbox(all_df,
                            lat="Lat",
                            lon="Long",
                            color="Temperature",
                            color_continuous_scale='rainbow',
                            zoom=5,
                            #size="Temperature",
                            height=1200,
                            width=1200)
    fig.update_layout(mapbox_style="carto-positron")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()


def plot_shifted(activity):
    d = {'Lat': activity.lat,
         'Lat_shifted': activity.lat + (activity.times - min(activity.times)) / 360000,
         'Long': activity.lon,
         'Long_shifted': activity.lon + (activity.times - min(activity.times)) / 360000,
         'Temperature': activity.temperature,
         'Speed': activity.speed*3.6}
    df = pd.DataFrame(data=d)
    fig = px.scatter_mapbox(df,
                            lat="Lat",
                            lon="Long_shifted",
                            color="Speed",
                            color_continuous_scale='rainbow',
                            zoom=15,
                            # size="Speed",
                            height=1200,
                            width=1200)
    fig.update_layout(mapbox_style="carto-positron")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()


def create_df(activities):
    fast = True
    if fast:
        dfs = []
    else:
        d = {'Lat': [],
             'Long': [],
             'Time': [],
             'Elevation': [],
             'Distance': [],
             'Duration': [],
             'Speed': [],
             'Heart Rate': [],
             'Power': [],
             'Name': [],
             'Year': [],
             'DateTime': [],
             'Virtual': [],
             'Type': [],
             'DT': [],
             'DS': [],
             'Power Zone': [],
             'Heart Rate Zone': [],
             'Cadence': [],
             'Accumulated Power': [],
             'Pedal Smoothness': [],
             'Torque Effectivenes': [],
             'Activity_Index': [],
             }
        all_df = pd.DataFrame(data=d)
    activity_names = []
    for idx, activity in enumerate(activities):
        activity_names.append(activity.name)
        if activity.sport =='':
            activity.sport = 'None'
        try:
            if activity.times.size >0:
                if fast:
                    dfs.append(create_single_df(activity,idx))
                else:
                    print(f'{activity.name =}, {idx = } ')
                    df = create_single_df(activity)
                    all_df = pd.concat([all_df, df])
        except Exception as error:
            print(f'Error: {activity.name =}')
            print(f'Exeption: {error =}')
            e = Exception
    if fast:
        all_df = pd.concat(dfs)
    with open("./data_frame.res", "wb") as fp:  # Pickling
        pickle.dump(all_df, fp)
    with open("./activity_names.res", "wb") as fp:  # Pickling
        pickle.dump(activity_names, fp)
    return all_df, activity_names


def create_single_df(activity, idx=1):
    d = {'Lat': activity.lat,
         'Long': activity.lon,
         'Time': activity.times,
         'Elevation': activity.elevation,
         'Distance': activity.distance,
         'Duration': activity.time_since_start,
         'Temperature': activity.temperature,
         'Heart Rate': activity.heart_rate,
         'Power': activity.power,
         'Speed': np.nan_to_num(activity.speed) * 3.6,
         'DateTime': activity.datetime,
         'DT': activity.dt,
         'DS': activity.ds,
         'Power Zone': activity.power_zone,
         'Heart Rate Zone': activity.heart_rate_zone,
         'Cadence': activity.cadence,
         'Accumulated Power': activity.accumulated_power,
         'Pedal Smoothness': activity.left_pedal_smoothness,
         'Torque Effectivenes': activity.left_torque_effectiveness,
         'Virtual': [activity.virtual] * activity.lat.size,
         'Name': [activity.name] * activity.lat.size,
         'Year': [activity.year] * activity.lat.size,
         'Type': [activity.sport] * activity.lat.size,
         'Activity_Index': [idx] * activity.lat.size,
         }
    df = pd.DataFrame(data=d)
    df['Heart Rate'].replace(255, np.nan, inplace=True)
    df['Temperature'].replace(127, np.nan, inplace=True)
    df['Heart Rate'] = df['Heart Rate'].astype('Int16')
    df['Year'] = df['Year'].astype('Int16')
    df['Power Zone'] = df['Power Zone'].astype('Int8')
    df['Heart Rate Zone'] = df['Heart Rate Zone'].astype('Int8')
    df['Cadence'] = df['Cadence'].astype('Int16')
    df['Power'] = df['Power'].astype('Int16')
    df['Temperature'] = df.Temperature.astype('Int8')
    df['Activity_Index'] = df.Activity_Index.astype('Int16')
    df.loc[df['Elevation'] < 0, 'Elevation'] = 0
    return df


def load_data_file(file="./activities.res"):
    with open(file, "rb") as fp:
        activities = pickle.load(fp)
    return activities


def load_activities_fit(files):
    activities =[]
    for file in files:
        activity = Activity()
        activity.read_fit(file)
        activities.append(activity)
    return activities


def load_Data_Strava_export():
    activities = []
    overview = pd.read_csv('./data/activities.csv')
    for idx, row in overview.iterrows():
        print(f'\tWorking on Activity {row["Activity Name"]} at the {row["Activity Date"]}, ID: {row["Activity ID"]}')
        activity = Activity()
        file = row['Filename']
        if isinstance(file,str):
            file = './data/' + file
            if not os.path.isfile(file.replace('.gz', '')) and file.__contains__('.gz'):
                with gzip.open(file, 'rb') as f_in:
                    with open(file.replace('.gz', ''), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            if os.path.isfile(file.replace('.gz', '')):
                if file.__contains__('.gpx'):
                    activity.load_file(file.replace('.gz', ''))
                elif file.__contains__('.fit'):
                    activity.read_fit(file.replace('.gz', ''))
                elif file.__contains__('.tcx'):
                    activity.read_tcx(file.replace('.gz', ''))
                else:
                    continue
                activity.sport = row['Activity Type']
                activity.name = row['Activity Name']
                if row['Commute']:
                    activity.sport = 'Commute ' + row['Activity Type']
                    activity.commute = True
                    activity.fix_duration(60)
                else:
                    activity.crop_activity(pd.to_datetime(row['Activity Date']), float(row['Elapsed Time']))
                if row['Activity Type'] == 'Virtual Ride':
                    activity.virtual = True
                activity.gear = row['Activity Gear']
                activities.append(activity)
            else:
                print(f'ERROR: File {file} does not exist')
        else:
            print(f'\tWARNING: File {file} not a string')
    activities.sort(key=operator.attrgetter('timestamp'))
    with open("./activities.res", "wb") as fp:  # Pickling
        pickle.dump(activities, fp)
    create_df(activities)
    for activity in activities:
        activity.reduce()
    with open("./activities_small.res", "wb") as fp:  # Pickling
        pickle.dump(activities, fp)
    return activities


def load_Data_turistveger():
    activities = []
    pathlist = pathlib.Path('./turistvegeer').glob('*.gpx')
    for idx, file in enumerate(pathlist):
        print(f'\tWorking on Activity {file} ')
        activity = Activity()
        file = './turistvegeer/' + file.name
        print(f'\tWorking on Activity {file} ')
        activity.load_file(file)
        activity.sport = 'Turistveger'
        activity.name = file.replace('.gpx', '')
        activities.append(activity)
    activities.sort(key=operator.attrgetter('timestamp'))
    with open("./turistveger.res", "wb") as fp:  # Pickling
        pickle.dump(activities, fp)
    d = {'Lat': activity.lat,
         'Long': activity.lon,
         'Time': activity.times,
         'Elevation': activity.elevation,
         'Distance': activity.distance,
         'Duration': activity.time_since_start,
         'Temperature': activity.temperature,
         'Type': [activity.sport] * activity.lat.size,
         'Heart Rate': activity.heart_rate,
         'Power': activity.power,
         'Speed': np.nan_to_num(activity.speed) * 3.6,
         'Virtual': [activity.virtual] * activity.lat.size,
         'Name': [activity.name] * activity.lat.size,
         'Year': [activity.year] * activity.lat.size,
         'DateTime': activity.datetime,
         'DT': activity.dt,
         'Power Zone': activity.power_zone,
         'Heart Rate Zone': activity.heart_rate_zone,
         'Index': [0] * activity.lat.size,
         }
    all_df = pd.DataFrame(data=d)

    for idx, activity in enumerate(activities):
        print(f'{activity.name =}, {idx = } ')
        try:
            if activity.times.size >0:
                df = create_single_df(activity)
                #df['Time'] = df['Time'].values.astype(dtype='datetime64[s]')
                all_df = pd.concat([all_df, df])
        except Exception as error:
            print(f'Error: {activity.name =}')
            print(f'Exeption: {error =}')
            e = Exception

    with open("./data_frame_turistveger.res", "wb") as fp:  # Pickling
        pickle.dump(all_df, fp)

    return activities


def calculate_eddington(df):
    df = df[df['Type'] != 'Alpine Ski'] ## Exclude Alpine Ski
    df = df[df['DateTime'] > datetime(year=2021, month=8, day=1)] # Exclude Nordkapp komoot data
    dfw = df[['DateTime','DS']]
    max_dist = int(np.max(df['Distance']))
    bins = np.linspace(1, max_dist, max_dist)
    results = np.zeros(max_dist)
    results_miles = np.zeros(max_dist)
    all_dates = {date.date() for date in df.DateTime}
    color = ['red'] * max_dist
    color_mile = ['orange'] * max_dist
    for date in sorted(all_dates):
        #dfy = df[[new_date.date() == date for new_date in df['DateTime']]]
        dfw = dfw[dfw['DateTime'] >= np.datetime64(date)]
        dfy = dfw[dfw['DateTime'] < np.datetime64(date + timedelta(days=1))]
        distance = np.sum(dfy.DS)
        print(f'{date = }, distance {int(distance)}')
        for indx, target_distance in enumerate(bins):
            if distance >= target_distance:
                results[indx] = results[indx] + 1
            else:
                break
        for indx, target_distance in enumerate(bins):
            if distance >= target_distance * 1.60934:
                results_miles[indx] = results_miles[indx] + 1
            else:
                break

    for indx, result in enumerate(results):
        if result > indx:
            color[indx] = 'green'
        else:
            print(f'Metric Eddington Number: {indx}')
            break

    eddington_number = 0
    for indx, result in enumerate(results_miles):
        if result > indx:
            color_mile[indx] = 'blue'
        else:
            print(f'Mile Eddington Number: {indx}')
            eddington_number = indx
            break

    with open("./eddington.res", "wb") as fp:  # Pickling
        pickle.dump([bins,results, color,results_miles, color_mile], fp)
    return eddington_number


def moving_mean(times, values, resolution):
    mean_times = []
    mean_values = []
    for indx, time in enumerate(times[resolution: times.size - resolution]):
        mean_times.append(time)
        mean_values_selection = values[indx: indx + (2*resolution+1)]
        mean_values.append(mean_values_selection.mean())
    mean_values = np.array(mean_values)
    mean_times = np.array(mean_times)
    return mean_times, mean_values

def read_dtm():
    filename = "./DEM/dtm10/data/dtm10_7002_2_10m_z33.tif"
    import rasterio
    from rasterio.plot import show
    fp = r'./DEM/dtm10/data/dtm10_7002_2_10m_z33.tif'
    dataset = rasterio.open(fp)
    dataset.bounds
    dataset.crs
    show(dataset)
    for val in dataset.sample([(290398.83243544295, 7019904.137248003)]):
        print(val)
    import utm
    utm.from_latlon(70,10,33)

def fix_elevation(activities):
    with open('./data_frame.res', "rb") as fp:
        df = pickle.load(fp)
    dff = extract_activities(df, activities, 'Append')
    a = []
    for ind in dff.index:
        lat = dff['Lat'][ind]
        long = dff['Long'][ind]
        a.append(fix_elevation_point(lat, long, dff['Elevation'][ind]))
    plt.plot(dff['Duration'], dff['Elevation'])
    plt.plot(dff['Duration'], a)
    plt.show()

def fix_elevation_point(lat, long, elevation):
    if lat is not None and long is not None:
        utm_values = utm.from_latlon(lat,long,33)
        bound = rasterio.coords.BoundingBox(utm_values[0],utm_values[1],utm_values[0],utm_values[1])
        found = False
        for dataset in datasets:
            bounds = dataset.bounds
            if not rasterio.coords.disjoint_bounds(bounds, bound):
                for val in dataset.sample([(utm_values[0],utm_values[1])]):
                    elevation = val[0]
                found = True
                #print('Found Point in Norway')
                break
        if not found:
            elevation = fix_elevation_hgt_point(lat, long, elevation)
    else:
        #print(f'{lat =}, {long = }, {elevation = }')
        elevation = None
    return elevation

def get_hgt_file_name(lat, lon):
    if lat > 0:
        lat_flag = 'N'
    else:
        lat = lat -1
        lat_flag = 'S'
    if lon > 0:
        lon_flag = 'E'
    else:
        lon = lon -1
        lon_flag = 'W'
    lat_name = format(abs(int(lat)), '02')
    lon_name = format(abs(int(lon)), '03')
    return lat_flag + lat_name+lon_flag+lon_name


SAMPLES = 3601 # Change this to 1201 for SRTM3
def read_elevation_from_file(hgt_file, lon, lat):
        # Each data is 16bit signed integer(i2) - big endian(>)
    elevations = buffered_nasa_sets[hgt_file]
    if lat > 0:
        lat_row = int(round((lat - int(lat)) * (SAMPLES - 1), 0))
    else:
        lat_row = int(round((lat - int(lat-1)) * (SAMPLES - 1), 0))
    if lon > 0:
        lon_row = int(round((lon - int(lon)) * (SAMPLES - 1), 0))
    else:
        lon_row = int(round((lon - int(lon-1)) * (SAMPLES - 1), 0))
    return elevations[SAMPLES - 1 - lat_row, lon_row].astype(int)


def fix_elevation_hgt_point(lat, long, elevation):
    if lat is not None and long is not None:
        hgt_file = './DEM/hgt_Files/' + get_hgt_file_name(lat, long) + '.hgt'
        if buffered_nasa_sets.__contains__(hgt_file):
            return read_elevation_from_file(hgt_file, long, lat)
        elif os.path.isfile(hgt_file):
            with open(hgt_file, 'rb') as hgt_data:
                elevations = np.fromfile(hgt_data, np.dtype('>i2'), SAMPLES * SAMPLES).reshape((SAMPLES, SAMPLES))
            print(f'Loading {hgt_file} into buffer')
            buffered_nasa_sets[hgt_file] = elevations
            return read_elevation_from_file(hgt_file, long, lat)
    return elevation


def fix_elevation_hgt(activities):
    with open('./data_frame.res', "rb") as fp:
        df = pickle.load(fp)
    dff = extract_activities(df, activities, 'Append')
    a = []
    for ind in dff.index:
        lat = dff['Lat'][ind]
        long = dff['Long'][ind]
        a.append(fix_elevation_hgt_point(lat, long, dff['Elevation'][ind]))
    plt.plot(dff['Duration'], a)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    activities = load_Data_Strava_export()
    Calculate_Polygons.analyse_activities()
    with open('./data_frame.res', "rb") as fp:
        df = pickle.load(fp)
    calculate_eddington(df)
    Tiles.analyse_dataframe(df)
