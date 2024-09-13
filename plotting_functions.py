import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import timezonefinder
import math
import pandas as pd
import pytz
import main
import plotly.express as px
import pickle


def get_plotting_dataset(df,activity_name, variable,
                 display, virtual, activity_type, append,
                 shift, start_date, end_date):

    if variable.__len__() >0:
        variable = variable[0]
    else:
        variable = 'Elevation'

    dff = main.get_subdataset(df, start_date, end_date, display, activity_name, append, virtual, activity_type)

    resolution = max(round(dff.shape[0] / 250000), 1)
    print(f'Resolution {resolution}')
    dff = dff.iloc[::resolution, :]
    if shift == 'Shifted':
        dff['Lat'] = dff['Duration'] / 60000 + dff['Lat']
        dff['Long'] = dff['Duration'] / 60000 + dff['Long']
    return dff


def update_x_timeseries(df, activity_name,yaxis_column_name,  axis_type, append,start_date,end_date, smoothing):
    if activity_name is not None:
        if activity_name.__len__() > 0:
            dff = df[df['Name'] == (activity_name[0])]
            for idx, name in enumerate(activity_name):
                dfn = df[df['Name'] == name]
                if idx>0:
                    if append == 'Append':
                        dfn['Duration'] = (dfn['Time'] - min(dff['Time']))/60
                        dfn['Distance'] = dfn['Distance'] + max(dff['Distance'])
                    elif append == 'Remove Gap':
                        dfn['Duration'] = dfn['Duration'] + max(dff['Duration'])
                        dfn['Distance'] = dfn['Distance'] + max(dff['Distance'])
                    dff = pd.concat([dff, dfn])
            if axis_type == 'DateTime':
                timezone_str = 'Europe/Oslo'
                if 'Zwift' not in activity_name[0]:
                    tf = timezonefinder.TimezoneFinder()
                    for index, row in dff.iterrows():
                        print(f'{row.Lat = },{row.Long = }')
                        if not math.isnan(row.Lat) and not math.isnan(row.Long):
                            timezone_str = tf.certain_timezone_at(lat=row.Lat, lng=row.Long)
                            print(f'{timezone_str = }')
                            break

                if timezone_str is None:
                    print("Could not determine the time zone")
                else:
                    # Display the current time in that time zone
                    timezone = pytz.timezone(timezone_str)
                    dt = dff.iloc[0].DateTime
                    print("The time in %s is %s" % (timezone_str, timezone.utcoffset(dt)))
                    dff['DateTime'] = dff['DateTime'] + timezone.utcoffset(dt)
        else:
            dff = df[df['Name'].isin(activity_name)]

    else:
        dff = df
    if start_date.__contains__('T'): start_date = start_date.split('T')[0]
    if end_date.__contains__('T'): end_date = end_date.split('T')[0]
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date) + timedelta(days=1)
    end.replace(hour=0, microsecond=0, second=0)

    dff = dff[dff['DateTime'] >= start]
    dff = dff[dff['DateTime'] <= end]

    return create_time_series(dff, axis_type, '', yaxis_column_name, smoothing)



def create_time_series(dff, axis_type, title, values, smoothing):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for idx, value in enumerate(values):
        if dff[value].dtype == np.dtype('O'):
            value = 'Elevation'
        if value =='Speed':
            print(f'{max(dff[value])}')
            print(f'{(dff[value])}')
        if value =='Power Zone':
            colors = [
                'blue' if i < 2 else 'green' if i < 3 else 'yellow' if i < 4 else 'orange' if i < 5 else 'red' for i in
                dff['Power Zone']]
            fig.add_trace(
                go.Scatter(x=dff[axis_type], y=dff['Power'], name=value, fillcolor='red',
                           marker=dict(color=colors, opacity=1),
                           mode='markers',),
                secondary_y=idx%2!=0
            )
            continue
        elif value =='Heart Rate Zone':
            colors = [
                'blue' if i < 2 else 'green' if i < 3 else 'yellow' if i < 4 else 'orange' if i < 5 else 'red'
                if i != None else 'black' for i in dff['Heart Rate Zone']]
            dff['Heart Rate Zones'] = colors
            dff['Heart Rate'] = dff['Heart Rate'].rolling(smoothing-1).mean()
            fig.add_trace(
                go.Scatter(x=dff[axis_type], y=dff['Heart Rate'], name=value, fillcolor='red',
                   customdata=dff[['Name', 'Heart Rate', 'Distance', 'Duration', 'Heart Rate Zones']],
                   mode='markers',
                   hovertemplate=
                        "<b>%{customdata[0]}</b><br>" +
                        "<b>Heart Rate: %{customdata[1]}</b><br><br>" +
                        "<b>Heart Rate Zone: %{customdata[4]}</b><br><br>" +
                        "Distance: %{customdata[2]:,.2f} km<br>" +
                        "Duration: %{customdata[3]:.2f} min<br>" +
                        "<extra></extra>",
                   marker=dict(color=colors, opacity=1)),
                   secondary_y=idx%2!=0,
            )
            continue
        elif value =='Elevation':
            dff['Altitude'] = dff[value].rolling(smoothing+30).mean()
            value = 'Altitude'
        elif value =='Calories':
            dff['Calories'] = dff['Accumulated Power']/862.6524312896405919661733615222

        for name in dff['Name'].unique():
            dfn = dff[dff['Name']==name]
            fig.add_trace(
                go.Scatter(x=dfn[axis_type],
                           y=dfn[value].rolling(smoothing-1).mean(),
                           mode='lines',
                           name=f'{name} {value}',
                           customdata=dfn[['Name', value, 'Distance', 'Duration']],
                           hovertemplate=
                                "<b>%{customdata[0]}</b><br>" +
                                "<b>"+value+": %{customdata[1]}</b><br><br>" +
                                "Distance: %{customdata[2]:,.2f} km<br>" +
                                "Duration: %{customdata[3]:.2f} min<br>" +
                                "<extra></extra>",),
                secondary_y=idx%2!=0,
            )
        fig.update_xaxes(title_text=axis_type)
        fig.update_layout(showlegend=False)


    fig.update_xaxes(showgrid=False)
    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)
    fig.update_layout(height=300, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
    fig.update_layout(showlegend=False)
    fig.update_layout(template='plotly_white')
    return fig


def update_overview_plots(df, activity_name, display, plot,  axis_type, start_date, end_date, append, virtual, activity_type):
    dffu = main.get_subdataset(df, start_date, end_date, display, activity_name, append, virtual, activity_type)
    dffu["DTh"] = dffu["DT"] / 3600
    dffu["DTm"] = dffu["DT"] / 60
    if axis_type == 'Distance':
        analysis_type = 'DS'
    else:
        analysis_type = 'DTh'
    if plot == 'Power Curve':
        curve = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0])
        curve_max = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0])
        curve_30D = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0])
        curve_90D = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0])
        labels = ['1 s','5 s','10 s','30 s','1 min','2 min','3 min','5 min','10 min','15 min','20 min','30 min','40 min','1 h', '90 min','2h','3h','4h','10h']
        activities = main.load_data_file("./activities_small.res")
        for activity in activities:
            curve_max = np.maximum(curve_max, activity.power_curve[0])
            if activity_name is not None:
                if activity_name.__contains__(activity.name):
                    curve = np.maximum(curve, activity.power_curve[0])
            if activity.timestamp > datetime.now().timestamp() -30*24*3600:
                curve_30D = np.maximum(curve_30D, activity.power_curve[0])
                if activity.timestamp > datetime.now().timestamp() -90*24*3600:
                    curve_90D = np.maximum(curve_90D, activity.power_curve[0])
        d = {'Times': np.hstack([labels,labels,labels,labels]),
             'Power': np.hstack([curve_max,curve_90D,curve_30D,curve]),
             'Activity' : np.hstack([
                                    ['Top']*curve.size,
                                    ['90 Days']*curve.size,
                                    ['30 Days'] * curve.size,
                                    ['Activity']*curve.size])
             }
        dfn = pd.DataFrame(data=d)
        fig = px.line(dfn, 'Times', 'Power',color='Activity')
        fig.update_layout(height=250, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
        fig.update_layout(template='plotly_white')
        return fig
    elif plot == 'Temperature':
        dffu = dffu.dropna(subset='Temperature')
        fig = px.histogram(dffu, x="Temperature", y=analysis_type, color="Name")
        fig.update_layout(showlegend=False, yaxis_title="Duration [h]")
        fig.update_layout(height=250, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
        fig.update_layout(template='plotly_white')
        return fig
    elif plot == 'Speed':
        dffu = dffu.dropna(subset='Speed')
        fig = px.histogram(dffu, x="Speed", y=analysis_type, color="Name")
        fig.update_layout(showlegend=False, yaxis_title="Duration [h]")
        fig.update_layout(height=250, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
        fig.update_layout(template='plotly_white')
        return fig
    elif plot == 'Elevation':
        dffu = dffu.dropna(subset='Elevation')
        fig = px.histogram(dffu, x="Elevation", y=analysis_type, color="Name")
        fig.update_layout(showlegend=False, yaxis_title="Duration [h]")
        fig.update_layout(height=250, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
        fig.update_layout(template='plotly_white')
        return fig
    elif plot == 'Heart Rate':
        max_hr = 185
        bin_edges = np.array([0.1,0.6,0.7,0.8,0.9,254/max_hr])*max_hr
        labels = ['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4', 'Zone 5']
        dffu = dffu.dropna(subset='Heart Rate')
        zones, bin_edges = np.histogram(dffu['Heart Rate'], bin_edges, weights=dffu['DTm'])
    elif plot == 'Power':
        labels = ['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4', 'Zone 5', 'Zone 6', 'Zone 7']
        ftp = 250
        bin_edges = np.array([0, 0.55,0.75,0.87,0.94,1.05,1.2,20])*ftp
        dffu = dffu.dropna(subset='Power')
        zones, bin_edges = np.histogram(dffu['Power'], bin_edges, weights=dffu['DTm'])
    elif plot == 'Time':
        bins = [0, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
        dffw = dffu.drop('Time_Bin', axis=1, errors='ignore')

        hours = dffw.DateTime.dt.hour
        tmp = pd.cut(hours, bins, right=False)

        dffw['Time_Bin'] = tmp
        grouped = dffw.groupby("Time_Bin")
        values = grouped[analysis_type].agg(np.sum).to_list()
        fig = go.Figure(data=[
            go.Bar(name='Times', x=bins[:-1], y=values),
        ])
        fig.update_layout(height=250, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
        fig.update_layout(template='plotly_white')
        return fig
    elif plot == 'Weekday':
        bins = [0, 1,2,3,4,5,6,7]
        bins_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thurstday', 'Friday', 'Saturday', 'Sunday']
        dffu = dffu.drop('Weekday Bin',axis=1, errors='ignore')
        dffu['Weekday Bin'] = pd.cut(dffu.DateTime.dt.day_of_week, bins, right=False)
        grouped = dffu.groupby("Weekday Bin")
        values = grouped[analysis_type].agg(np.sum).to_list()
        fig = go.Figure(data=[
            go.Bar(name='Times', x=bins_labels, y=values),
        ])
        fig.update_layout(height=250, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
        fig.update_layout(template='plotly_white')
        return fig
    elif plot == 'Month':
        bins = [1,2,3,4,5,6,7,8,9,10,11,12,13]
        bins_labels = ['January', 'February', 'March', 'April', 'May', 'June', 'July','August','September','October','November','December']
        dffu.drop('Month Bin', errors='ignore')
        dffu['Month Bin'] = pd.cut(dffu.DateTime.dt.month, bins, right=False)
        data = []
        for year in dffu['Year'].unique():
            dffy = dffu[dffu['Year'] == year]
            grouped = dffy.groupby("Month Bin")
            values = grouped[analysis_type].agg(np.sum).to_list()
            data.append(go.Bar(name=int(year), x=bins_labels, y=values))
        fig = go.Figure(data=data)
        fig.update_layout(height=250, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
        fig.update_layout(template='plotly_white')
        return fig
    elif plot == 'Week':
        bins = np.linspace(1,54,54)
        bins_labels = np.linspace(1,53,53)
        dffu.drop('Week Bin', errors='ignore')
        dffu['Week Bin'] = pd.cut(dffu.DateTime.dt.isocalendar().week, bins, right=False)
        data = []
        for year in dffu['Year'].unique():
            dffy = dffu[dffu['Year'] == year]
            grouped = dffy.groupby("Week Bin")
            values = grouped[analysis_type].agg(np.sum).to_list()
            data.append(go.Bar(name=int(year), x=bins_labels, y=values))
        fig = go.Figure(data=data)
        fig.update_layout(height=250, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
        fig.update_layout(template='plotly_white')
        return fig
    elif plot == 'Eddington Number':
        with open("./eddington.res", "rb") as fp:  # Pickling
            res = pickle.load(fp)
            bins = res[0]
            results = res[1]
            color = res[2]
            results_miles = res[3]
            color_mile= res[4]
        data = []
        data.append(go.Bar(name='km', x=bins, y=results, marker_color=color))
        data.append(go.Bar(name='Miles', x=bins, y=results_miles, marker_color=color_mile))
        fig = go.Figure(data=data)
        fig.update_layout(height=250, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
        fig.update_layout(template='plotly_white',bargap = 0.0)
        fig.update_layout(xaxis_range=[50, 100])
        fig.update_layout(yaxis_range=[0, 150])
        return fig

    elif plot == 'Tiles':
        with open("./Tiles.res", "rb") as fp:  # Pickling
            all_Tiles = pickle.load(fp)
        bins = [2,3,4,5,6,7,8,9,10,11,12,13,14]
        results = []
        for bin in bins:
            tiles = all_Tiles[bin]
            results.append(tiles.__len__())
        data = []
        data.append(go.Bar(name='Number', x=bins, y=results))
        #data.append(go.Bar(name='Miles', x=bins, y=results_miles, marker_color=color_mile))
        fig = go.Figure(data=data)
        fig.update_layout(height=250, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
        fig.update_layout(template='plotly_white',bargap = 0.0)
        #fig.update_layout(xaxis_range=[2, 14])
        return fig
    elif plot == 'Tile Area':
        with open("./Tiles.res", "rb") as fp:  # Pickling
            all_Tiles = pickle.load(fp)
        bins = [2,3,4,5,6,7,8,9,10,11,12,13,14]
        results = []
        for bin in bins:
            tiles = all_Tiles[bin]
            results.append(tiles.__len__()/(4**bin)*100)
        data = []
        data.append(go.Bar(name='Visited Ratio [%]', x=bins, y=results))
        #data.append(go.Bar(name='Miles', x=bins, y=results_miles, marker_color=color_mile))
        fig = go.Figure(data=data)
        fig.update_layout(height=250, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
        fig.update_layout(template='plotly_white',bargap = 0.0)
        #fig.update_layout(xaxis_range=[2, 14])
        return fig
    d = {'Zones': labels, 'Duration': zones}
    dfn = pd.DataFrame(data=d)
    fig = px.bar(dfn,'Zones', 'Duration')
    fig.update_layout(height=250, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
    fig.update_layout(template='plotly_white')
    return fig
