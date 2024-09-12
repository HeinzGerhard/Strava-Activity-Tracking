import numpy as np

from main import *
from dash import Dash, html, dcc, Input, Output, State,  callback, Patch
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pathlib
import pytz
import timezonefinder
import scipy
import statshunters_import
import Tiles

from flask import Flask

access_token = open(".mapbox_token").read()
px.set_mapbox_access_token(access_token)

with open('./data_frame.res', "rb") as fp:
    df = pickle.load(fp)
df.loc[df.Speed > 200, 'Speed'] = 0 ## Remove 232 Values from Strava Iphone
with open('./activity_names.res', "rb") as fp:
    activity_names = pickle.load(fp)

with open('./data_frame_turistveger.res', "rb") as fp:
    df_turistveger = pickle.load(fp)
#df = df.iloc[::5, :]

update =False

names = df['Name'].unique()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

server = Flask(__name__)
app = Dash(server=server, external_stylesheets=external_stylesheets)

colorscales = px.colors.named_colorscales()
#app = Dash(__name__, external_stylesheets=external_stylesheets)

#df = pd.read_csv('https://plotly.github.io/datasets/country_indicators.csv')

daterange = pd.date_range(start='2021',end='2024',freq='W')
dff = df.iloc[::23, :]
fig = px.scatter_mapbox(dff,
                        lat="Lat",
                        lon="Long",
                        color='Elevation',
                        color_continuous_scale='rainbow',
                        height=550
                        # hover_name=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name']
                        )
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
names = dff['Name'].unique()


app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Dropdown(
                options=names,
                value=[names[-1]],
                id='crossfilter-activity-name',
                multi=True
            ),
            html.Div([dcc.RadioItems(
                ['Single', 'All'],
                'All',
                id='crossfilter-display-selection',
                labelStyle={'display': 'inline-block', 'marginTop': '10px'}
            )], style={'width': '20%', 'display': 'inline-block'}),
            html.Div([dcc.RadioItems(
                ['All', 'Outdoors', 'Virtual'],
                'All',
                id='crossfilter-virtual',
                labelStyle={'display': 'inline-block', 'marginTop': '5px'}
            )], style={'width': '30%', 'display': 'inline-block'}
            ),
            html.Div([dcc.RadioItems(
                ['Append', 'Remove Gap', 'Seperate'],
                'Append',
                id='crossfilter-append',
                labelStyle={'display': 'inline-block', 'marginTop': '5px'}
            )], style={'width': '50%', 'display': 'inline-block'}
            )
        ],
        style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            html.Div(
                [dcc.Dropdown(
                #df.columns,
                    ['Elevation', 'Speed', 'Distance', 'Duration', 'Temperature', 'Heart Rate', 'Power','Cadence',
                     'Accumulated Power', 'Calories', 'Pedal Smoothness', 'Torque Effectivenes', 'Name', 'Year',
                     'Type', 'Time', 'Lat', 'Long', 'Power Zone', 'Heart Rate Zone'],
                    value=['Elevation'],
                    id='crossfilter-yaxis-column',
                    multi=True
                )], style={'width': '99%', 'display': 'inline-block', 'float': 'right'}),
            html.Div([
                html.Div([dcc.Dropdown(
                    #df.columns,
                        ['Duration', 'Distance', 'DateTime'],
                        'Distance',
                        id='crossfilter-yaxis-type',
                    )], style={'width': '49%', 'display': 'inline-block'}),
                html.Div([dcc.Dropdown(
                    ['Temperature', 'Heart Rate', 'Power',
                     'Power Curve', 'Speed', 'Elevation',
                     'Time', 'Weekday', 'Month', 'Week',
                     'Eddington Number', 'Tiles', 'Tile Area'],
                    'Eddington Number',
                    id='crossfilter-overview-plot'
                )], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}
                )
            ], style={'width': '99%', 'float': 'right', 'display': 'inline-block'})
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'padding': '10px 5px'
    }),


    html.Div([
        dcc.Graph(
            figure=fig,
            id='crossfilter-indicator-scatter',
            #hoverData={'points': [{'customdata': 'Japan'}]}
        )
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
    html.Div([
        dcc.Graph(id='x-time-series'),
        dcc.Graph(id='y-time-series'),
    ], style={'display': 'inline-block', 'width': '49%'}),

    html.Div([
        dcc.Dropdown(
            options=df['Type'].unique(),
            value=df['Type'].unique(),
            id='crossfilter-activity-type',
            multi=True
        )]),


    html.Div([
        html.Div([
            dcc.DatePickerRange(
                    id='my-date-picker-range',
                    start_date=min(df['DateTime']),
                    end_date=max(df['DateTime']),
                    #initial_visible_month=max(df['DateTime'])-timedelta(days=120),
                    display_format='D/M/YY',
                    style={'width': '100%'},
                    show_outside_days=False,
                    number_of_months_shown=5,
                    stay_open_on_select=True)
        ], style={'width': '42%', 'display': 'inline-block'}),
        html.Div([
            dcc.RadioItems(
                    ['Normal', 'Shifted'],
                    'Normal',
                    id='crossfilter-shift',
                    labelStyle={'display': 'inline-block', 'marginTop': '5px'},
                    #style={'width': '50%', 'float': 'right'}
        )], style={'width': '22%', 'display': 'inline-block'}),
        html.Div([
            dcc.Upload(
                    id='crossfilter-upload'
                ,children=html.Div([
            'Drag or ',
            html.A('Select Files')
        ]),
        style={
            'width': '90%',
            'height': '30px',
            'lineHeight': '30px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '5px'
        },
        )], style={'width': '22%', 'display': 'inline-block'}),
        html.Div([
                    html.Button('Reload', id='btn', n_clicks=0)], style={'width': '10%', 'display': 'inline-block'}),
    ], style={'width': '49%', 'display': 'inline-block'}),
    html.Div([
        html.Div([
        dcc.Dropdown(
            options=["open-street-map", "carto-positron", "carto-darkmatter", "basic", "streets", "outdoors", "light", "dark", "satellite", "satellite-streets",
                     "stamen-terrain", "stamen-toner", "stamen-watercolor"],
            value="carto-positron",
            id='crossfilter-map-type',
            #style={'width': '50%', 'display': 'inline-block'}
        )], style={'width': '21%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                options=colorscales,
                value="rainbow",
                id='crossfilter-color-map',
                #style={'width': '50%', 'display': 'inline-block'}
        )], style={'width': '20%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                options=["point", "line", "heatmap", "hexagon_heatmap", 'turistveger', 'polygon', 'tiles', 'Statshunter tiles','Activity tiles'],
                value="point",
                id='crossfilter-map-plot',
                #style={'width': '50%', 'float': 'right', 'display': 'inline-block'}
        )], style={'width': '23%', 'display': 'inline-block'}),
        html.Div([
            dcc.Slider(1, 300, 25, value=1,
                id='crossfilter-smoothing',
        )], style={'width': '32%', 'display': 'inline-block'}),
    ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'}),
])

@callback(
    Output('crossfilter-indicator-scatter', 'figure',allow_duplicate=True),
    Input('crossfilter-activity-name', 'value'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-display-selection', 'value'),
    Input('crossfilter-virtual', 'value'),
    Input('crossfilter-activity-type','value'),
    Input('crossfilter-append','value'),
    State('crossfilter-map-type', 'value'),
    Input('crossfilter-map-plot', 'value'),
    Input('crossfilter-shift','value'),
    State('crossfilter-color-map','value'),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date'),
    Input('crossfilter-smoothing', 'value'),prevent_initial_call=True)
def update_graph(activity_name, yaxis_column_name,
                 display, virtual, activity_type,append,
                 map_type,
                 map_plot,shift,color_map,start_date,end_date, smoothing):
    print('Old Update Graph')
    global df
    global names
    if yaxis_column_name.__len__() >0:
        yaxis_column_name = yaxis_column_name[0]
    else:
        yaxis_column_name = 'Elevation'

    dff = get_subdataset(df, start_date, end_date, display, activity_name, append, virtual, activity_type)

    resolution = max(round(dff.shape[0] / 250000),1)
    print(f'Resolution {resolution}')
    dff = dff.iloc[::resolution, :]
    data = []
    if shift == 'Shifted':
        dff['Lat'] = dff['Duration']/ 60000 + dff['Lat']
        dff['Long'] = dff['Duration']/ 60000 + dff['Long']
    if map_plot == 'hexagon_heatmap':
        dff['Lat'] = dff['Lat'].apply(pd.to_numeric)
        dff['Long'] = dff['Long'].apply(pd.to_numeric)
        fig = ff.create_hexbin_mapbox(
            data_frame=dff, lat="Lat", lon="Long",
            nx_hexagon=300, opacity=0.5, #labels={"color": "Point Count"},turbo
            min_count=1, color_continuous_scale="turbo",
            show_original_data=True,
            mapbox_style=map_type,
            original_data_marker=dict(size=2, opacity=0.6, color="deeppink")
        )
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        fig.update_layout(showlegend=False)
        return fig
    elif map_plot == 'line':
        if display == 'All' or activity_name.__len__() == 0:
            dff.loc[dff['Distance'] < 0.15, 'Lat'] = None
            dff.loc[dff['Distance'] < 0.15, 'Long'] = None
        else:
            dff.loc[dff['Distance'] < 0.01, 'Lat'] = None
            dff.loc[dff['Distance'] < 0.01, 'Long'] = None

        fig = px.line_mapbox(dff,
                             lat="Lat",
                             lon="Long")

        fig.update_layout(mapbox_zoom=14,
                          margin={"r": 0, "t": 0, "l": 0, "b": 0})

        return fig
    elif map_plot == 'heatmap':
        fig = go.Figure(go.Densitymapbox(lat=dff['Lat'], lon=dff['Long'], #z=dff['Speed'],
                                         radius=int(smoothing/10)+1))
        fig.update_layout(mapbox_style=map_type)
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        return fig

    elif map_plot == 'turistveger':
        dff = pd.concat([dff, df_turistveger])
        fig = px.scatter_mapbox(dff,
                                lat="Lat",
                                lon="Long",
                                color=yaxis_column_name,
                                color_continuous_scale=color_map,
                                height=550
                                # hover_name=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name']
                                )
        fig.update_layout(mapbox_style=map_type)
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        return fig

    elif map_plot == 'polygon':
        with open("./polygons.res", "rb") as fp:
            polygons_list = pickle.load(fp)
        with open("./activity_polygons.res", "rb") as fp:
            activity_polygons = pickle.load(fp)
        import geopandas as gpd
        for name in activity_name:
            for polygon in activity_polygons[name]:
                polygons_list.append(polygon)
        polygons_list.reverse()
        gdf = gpd.GeoSeries(polygons_list)
        gdf.crs = "epsg:4326"
        areas = gdf.to_crs({'init': 'epsg:32633'}) \
            .map(lambda p: p.area / 10 ** 6)
        gdf = gpd.GeoDataFrame(gdf)
        gdf = gdf.assign(area=areas)
        gdf = gdf.set_geometry(0)

        fig = px.choropleth_mapbox(gdf,
                           geojson=gdf.geometry,
                           locations=gdf.index,
                           color="area",
                           center={"lat": 64, "lon": 9},
                           mapbox_style=map_type,
                           opacity=0.5,
                           )
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        return fig
    elif map_plot == 'tiles':

        zoom = 6
        zoom = int(smoothing/25+2)
        print(f'{zoom = }')

        long, lat = Tiles.generate_reduced_lines(zoom)

        data.append(go.Scattermapbox(
            lon=long,
            lat=lat,
            mode='lines',
            fillcolor='red',
            line=dict(color='red', width=0.5),
            name='All-Lines', opacity=.25
        ))

        with open('Tiles.res', "rb") as fp:
            all_Tiles = pickle.load(fp)
        #tiles = Tiles.check_tiles(zoom,df)
        tiles = all_Tiles[zoom]
        x = []
        y = []
        for tile in tiles:
            lon, lat = Tiles.tile_outline(tile, zoom)
            x.append(None)
            y.append(None)
            x.extend(lon)
            y.extend(lat)

        data.append(go.Scattermapbox(
            lon=x,
            lat=y,
            mode='lines',
            fillcolor='red',
            line=dict(color='red'),
            name='all'
        ))

        data.append(go.Scattermapbox(
            lon=dff['Long'],
            lat=dff['Lat'],
            mode='markers',
            name='Trace',
            marker=go.scattermapbox.Marker(color=list(dff[yaxis_column_name]),
                        size=5,
                        colorscale=color_map,  # one of plotly colorscales
                        showscale=True,
                        #text=dff['Name']
                        )
        ))
        fig = go.Figure(data=data)
        fig.update_layout(margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
                          mapbox={
                              'accesstoken':open(".mapbox_token").read(),
                              #'center': {'lon': 139, 'lat': 36.5},
                              'style': map_type,
                              #'zoom': 4.5
                          },
                          #width=1600,
                          height=550,
                           showlegend=False)
        return fig

    elif map_plot == 'Statshunter tiles':
        with open('stathunters/results.res', "rb") as fp:
            output = pickle.load(fp)
        tiles = output[0]
        statshunter_activities = output[1]
        x = []
        y = []
        for tile in tiles:
            lon, lat = statshunters_import.tile_outline(tile)
            x.append(None)
            y.append(None)
            x.extend(lon)
            y.extend(lat)

        data.append(go.Scattermapbox(
            lon=x,
            lat=y,
            mode='lines',
            fillcolor='red',
            line=dict(color='green'),
            name='all'
        ))
        for name in activity_name:
            activity = statshunter_activities[name]
            for tile in activity['tiles']:
                uid = "{0}_{1}".format(tile['x'], tile['y'])
                lon, lat = statshunters_import.tile_outline(uid)
                data.append(go.Scattermapbox(
                    lon=lon,
                    lat=lat,
                    mode='lines',
                    fillcolor='red',
                    line=dict(color='red'),
                    name=uid
                ))
        data.append(go.Scattermapbox(
            lon=dff['Long'],
            lat=dff['Lat'],
            mode='markers',
            name='Trace',
            marker=go.scattermapbox.Marker(color=list(dff[yaxis_column_name]),
                        size=5,
                        colorscale=color_map,  # one of plotly colorscales
                        showscale=True,
                        #text=dff['Name']
                        )
        ))
        fig = go.Figure(data=data)
        fig.update_layout(margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
                          mapbox={
                              'accesstoken':open(".mapbox_token").read(),
                              #'center': {'lon': 139, 'lat': 36.5},
                              'style': map_type,
                              'zoom': 4.5},
                          #width=1600,
                          height=550,
                           showlegend=False)
        return fig
    elif map_plot == 'Activity tiles':
        zoom = 6
        zoom = int(smoothing/25+2)
        print(f'{zoom = }')

        with open('Tiles.res', "rb") as fp:
            all_Tiles = pickle.load(fp)
        #tiles = Tiles.check_tiles(zoom,df)
        tiles = all_Tiles[zoom]
        x = []
        y = []
        for tile in tiles:
            lon, lat = Tiles.tile_outline(tile, zoom)
            x.append(None)
            y.append(None)
            x.extend(lon)
            y.extend(lat)

        data.append(go.Scattermapbox(
            lon=x,
            lat=y,
            mode='lines',
            fillcolor='red',
            line=dict(color='green'),
            name='all'
        ))


        tiles = Tiles.check_tiles(zoom,dff)
        x = []
        y = []
        for tile in tiles:
            lon, lat = Tiles.tile_outline(tile,zoom)
            x.append(None)
            y.append(None)
            x.extend(lon)
            y.extend(lat)

        data.append(go.Scattermapbox(
            lon=x,
            lat=y,
            mode='lines',
            fillcolor='red',
            line=dict(color='red'),
            name='all'
        ))
        data.append(go.Scattermapbox(
            lon=dff['Long'],
            lat=dff['Lat'],
            mode='markers',
            name='Trace',
            marker=go.scattermapbox.Marker(color=list(dff[yaxis_column_name]),
                        size=5,
                        colorscale=color_map,  # one of plotly colorscales
                        showscale=True,
                        #text=dff['Name']
                        )
        ))
        fig = go.Figure(data=data)
        fig.update_layout(margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
                          mapbox={
                              'accesstoken':open(".mapbox_token").read(),
                              #'center': {'lon': 139, 'lat': 36.5},
                              'style': map_type,
                              #'zoom': 4.5
                          },
                          #width=1600,
                          height=550,
                           showlegend=False)
        return fig
    else:
        dff['Calories'] = dff['Accumulated Power'] / 862.6524312896405919661733615222
        if not df[yaxis_column_name].dtype == np.dtype('O'):
            values = np.array(dff[yaxis_column_name].astype(float))

            #print(f'{values = }')
            values = values[~np.isnan(values)]
            color_range = [min(values),max(values)]
            range_limits = [
                ['Power', 0, 350],
                ['Temperature', -273.15, 40],
                ['Heart Rate', 40, 200],
                ['Duration', 0, 20*60],
            ]
            for limit in range_limits:
                if yaxis_column_name == limit[0]:
                    print(f'{limit[1] = }, {limit[2] = }, {color_range[0] = }, {color_range[1] = }')
                    color_range = [
                        max(color_range[0],limit[1]),
                        min(color_range[1],limit[2])
                    ]
            dff[yaxis_column_name] = dff[yaxis_column_name].rolling(smoothing).mean()
            fig = px.scatter_mapbox(dff,
                                    lat="Lat",
                                    lon="Long",
                                    color=yaxis_column_name,
                                    color_continuous_scale=color_map,
                                    range_color = color_range,
                                    height=550,
                                    mapbox_style=map_type,
                                    hover_name='Name'
                                    )
        else:
            fig = px.scatter_mapbox(dff,
                                    lat="Lat",
                                    lon="Long",
                                    color=yaxis_column_name,
                                    height=550,
                                    mapbox_style=map_type,
                                    hover_name='Name'
                                    )
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        return fig

        print('Go Figure')
        print(f'{yaxis_column_name =}')

        data.append(go.Scattermapbox(
            lon=dff['Long'],
            lat=dff['Lat'],
            mode='markers',
            marker=go.scattermapbox.Marker(color=list(dff[yaxis_column_name]),
                        size=5,
                        colorscale=color_map,  # one of plotly colorscales
                        showscale=True
                        )
        ))
        fig = go.Figure(data=data)
        fig.update_layout(margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
                          mapbox={
                              'accesstoken':open(".mapbox_token").read(),
                              #'center': {'lon': 139, 'lat': 36.5},
                              'style': map_type,
                              'zoom': 4.5},
                          #width=1600,
                          height=550,
                          coloraxis_colorbar=dict(
                                title="Your Title",
                                ),
                          )
        return fig

def create_time_series(dff, axis_type, title, values, smoothing):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for idx, value in enumerate(values):
        if df[value].dtype == np.dtype('O'):
            value = 'Elevation'
        if value =='Speed':
            print(f'{max(dff[value])}')
            print(f'{(dff[value])}')
        if value =='Power Zone':
            colors = [
                'blue' if i < 2 else 'green' if i < 3 else 'yellow' if i < 4 else 'orange' if i < 5 else 'red' for i in
                dff['Power Zone']]
            #fig = px.scatter(dff, x=axis_type, y='Power', color=value)
            fig.add_trace(
                go.Scatter(x=dff[axis_type], y=dff['Power'], name=value, fillcolor='red'),
                secondary_y=idx%2!=0,
            )
            fig.update_traces(mode='markers')
            continue
        elif value =='Heart Rate Zone':
            colors = [
                'blue' if i < 2 else 'green' if i < 3 else 'yellow' if i < 4 else 'orange' if i < 5 else 'red'
                if i != None else 'black' for i in dff['Heart Rate Zone']]
            dff['Heart Rate Zones'] = colors
            dff['Heart Rate'] = dff['Heart Rate'].rolling(smoothing).mean()
            #fig = px.scatter(dff, x=axis_type, y='Heart Rate', color=value)
            fig.add_trace(
                go.Scatter(x=dff[axis_type], y=dff['Heart Rate'], name=value, fillcolor='red'),
                secondary_y=idx%2!=0,
            )
            fig.update_traces(mode='markers')
            continue
        elif value =='Elevation':
            dff['Altitude'] = dff[value].rolling(smoothing+30).mean()
            value = 'Altitude'
        elif value =='Calories':
            dff['Calories'] = dff['Accumulated Power']/862.6524312896405919661733615222
            #fig = px.scatter(dff, x=axis_type, y='Altitude', color='Name')

        for name in dff['Name'].unique():
            dfn = dff[dff['Name']==name]
            fig.add_trace(
                go.Scatter(x=dfn[axis_type], y=dfn[value].rolling(smoothing).mean(), name=f'{name} {value}'),
                secondary_y=idx%2!=0,
            )
        # Set x-axis title
        fig.update_xaxes(title_text=axis_type)
        fig.update_layout(showlegend=False)
        #return fig

        #fig = px.scatter(dff, x=axis_type, y=value, color='Name')
        #fig.update_traces(mode='lines')
    fig.update_xaxes(showgrid=False)
    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)
    fig.update_layout(height=300, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
    fig.update_layout(showlegend=False)
    fig.update_layout(template='plotly_white')
    return fig

def create_time_series_smooth(dff, axis_type, title, value):
    fig = go.Figure()

    for activity_name in dff['Name'].unique():

        dfa = dff[dff['Name'] == activity_name]
        print(f'{value = }')
        print(f"{value == 'Heart Rate' = }")
        if value == 'Heart Rate':
            fig.add_trace(go.Scatter(x=dfa[axis_type], y=scipy.signal.savgol_filter(dfa[value], 51, 3),
                                     name=activity_name+"smoothed",
                                     #text=["tweak line smoothness<br>with 'smoothing' in line object"],
                                     #hoverinfo='name',
                                     line_shape='spline'))
        else:
            fig.add_trace(go.Scatter(x=dfa[axis_type], y=dfa[value], color=activity_name,
                                     #text=["tweak line smoothness<br>with 'smoothing' in line object"],
                                     #hoverinfo='text+name',
                                     #hoverinfo='name',
                                     #hoverinfo=dfa[axis_type],
                                     line_shape='spline'))

    fig.update_layout(showlegend=False)
    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)
    fig.update_layout(height=300, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
    fig.update_layout(template='plotly_white')
    return fig


@callback(
    Output('x-time-series', 'figure'),
    Input('crossfilter-activity-name', 'value'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-yaxis-type', 'value'),
    Input('crossfilter-append','value'),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date'),
    Input('crossfilter-smoothing', 'value'),)
def update_x_timeseries(activity_name,yaxis_column_name,  axis_type, append,start_date,end_date, smoothing):
    global df
    if activity_name is not None:
        if activity_name.__len__() > 0:
            dff = df[df['Name'] == (activity_name[0])]
            for idx, name in enumerate(activity_name):
                dfn = df[df['Name'] == name]
                if idx>0:
                    if append == 'Append':
                        dfn['Duration'] = (dfn['Time'] - min(dff['Time']))/60
                        dfn['Distance'] = dfn['Distance'] + max(dff['Distance'])
                    elif append == 'No Gap':
                        dfn['Duration'] = dfn['Duration'] + max(dff['Duration'])
                        dfn['Distance'] = dfn['Distance'] + max(dff['Distance'])
                    dff = pd.concat([dff, dfn])
            if axis_type == 'DateTime':
                timezone_str = 'Europe/Oslo'
                if 'Zwift' not in activity_name[0]:
                    tf = timezonefinder.TimezoneFinder()
                    #print(f'{dff.iloc[0].Lat},{dff.iloc[0].Long}')
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


@callback(
    Output('y-time-series', 'figure'),
    Input('crossfilter-activity-name', 'value'),
    Input('crossfilter-display-selection', 'value'),
    Input('crossfilter-overview-plot', 'value'),
    Input('crossfilter-yaxis-type', 'value'),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date'),
    Input('crossfilter-append','value'),
    Input('crossfilter-virtual', 'value'),
    Input('crossfilter-activity-type', 'value'))
def update_overview_plots(activity_name, display, plot,  axis_type, start_date, end_date, append, virtual, activity_type):
    global df
    dffu = get_subdataset(df, start_date, end_date, display, activity_name, append, virtual, activity_type)
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
        activities = load_data_file("./activities_small.res")
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


@app.callback(
    Output('crossfilter-activity-name', 'options', allow_duplicate=True),
    Input('crossfilter-virtual', 'value'),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date'),
    Input('crossfilter-activity-type', 'value'),
    prevent_initial_call=True
)
def update_date_selector(virtual,start_date, end_date, type):
    global df
    if start_date.__contains__('T'): start_date = start_date.split('T')[0]
    if end_date.__contains__('T'): end_date = end_date.split('T')[0]
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date) + timedelta(days=1)
    end.replace(hour=0, microsecond=0, second=0)

    print(f'{end_date = }')
    print(f'{start_date = }')
    dff = df[df['DateTime'] >= start]
    dff = dff[dff['DateTime'] <= end]
    if virtual == 'Outdoors':
            dff = dff[dff['Virtual'] == False]
    elif virtual == 'Virtual':
            dff = dff[dff['Virtual'] == True]
    #        print(f'virtual  {dff.size}')
    else:
            print(f'All {dff.size}')
    #
    dff = dff[dff['Type'].isin(type)]
    options = []
    for name in dff['Name'].unique():
            options.append({'label': name, 'value': name})

    return np.flip(dff['Name'].unique())

def get_data_subset(df, activity_name, start, end):
    pass

@app.callback(
    Output('crossfilter-activity-name', 'options',allow_duplicate=True),
    Input('crossfilter-upload', 'filename'),
    prevent_initial_call=True
)
def upload(filename):
    print(f'{filename = }')
    global df
    if filename is not None:
        activities = load_data_file()
        if filename.__contains__('.gpx'):
            print('gpx')
            activity=Activity()
            path=pathlib.Path('C:/Users/nicol/Downloads')
            activity.load_file(path / filename)
            activities.append(activity)
        elif filename.__contains__('.tcx'):
            print('tcx')
            activity=Activity()
            path=pathlib.Path('C:/Users/nicol/Downloads')
            activity.read_tcx(path / filename)
            activities.append(activity)

        else:
            print('fit')
            activity = Activity()
            path = pathlib.Path('C:/Users/nicol/Downloads')
            activity.read_fit(path / filename)

        dfn = create_single_df(activity, np.max(df['Activity_Index'].unique())+1)
        df = pd.concat([df, dfn])
        with open("./data_frame.res", "wb") as fp:  # Pickling
            pickle.dump(df, fp)
        activities.append(activity)
        with open("./activities.res", "wb") as fp:  # Pickling
            pickle.dump(activities, fp)
        for activity in activities:
            activity.reduce()
        with open("./activities_small.res", "wb") as fp:  # Pickling
            pickle.dump(activities, fp)
        del activities
    print('Done')
    return np.flip(df['Name'].unique())

@app.callback(
    Output('crossfilter-activity-name', 'options',allow_duplicate=True),
    Input('btn', 'n_clicks'),
    prevent_initial_call=True
)
def update_Button(btn):
    global df
    global update
    if update:
        print('Already Updating')
    else:
        update = True
        activities = load_Data_Strava_export()
        df = create_df(activities)
        del activities
        update = False
    return np.flip(df['Name'].unique())


@app.callback(
    Output('crossfilter-indicator-scatter', 'figure', allow_duplicate=True),
    Input('crossfilter-map-type', 'value'),
    prevent_initial_call=True
)
def update_Map_type(map):
    print(f'Update to map {map}')
    patched_figure = Patch()
    patched_figure.layout.mapbox.style = map
    return patched_figure


@app.callback(
    Output('crossfilter-indicator-scatter', 'figure', allow_duplicate=True),
    Input('crossfilter-color-map', 'value'),
    prevent_initial_call=True
)
def update_colorbar(colorbar):
    print(f'Update colorscale {colorbar}')
    patched_figure = Patch()
    patched_figure.layout.colorscale = colorbar
    patched_figure.layout.colorbar = colorbar
    #patched_figure.data.colorbar = colorbar
    #patched_figure.marker.colorbar = colorbar
    print(f' {patched_figure.data}')
    print(f' {patched_figure.data.colorbar}')
    print(f' {patched_figure.marker}')
    print(f' {patched_figure.marker.colorbar}')
    #patched_figure.data.colorbar = [colorbar]
    #patched_figure.data.marker.colorbar = colorbar
    print(f' Finished Update colorscale {patched_figure}')
    return patched_figure



if __name__ == '__main__':
    #app.run(debug=True, use_reloader=False, port=12348)
    app.run_server(host='0.0.0.0',debug=False, port=8052)