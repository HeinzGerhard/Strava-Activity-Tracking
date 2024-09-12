import numpy as np
from dash import Dash, html, dcc, Input, Output, State, callback, Patch
import plotly.graph_objects as go
import Tiles
import json
import plotting_functions
import geopandas as gpd
import plotly.express as px
import pickle
import pandas as pd

from flask import Flask

access_token = open(".mapbox_token").read()
px.set_mapbox_access_token(access_token)

with open('./data_frame.res', "rb") as fp:
    df = pickle.load(fp)
df.loc[df.Speed > 200, 'Speed'] = 0  ## Remove 232 Values from Strava Iphone

with open('./activity_names.res', "rb") as fp:
    activity_names = pickle.load(fp)

with open('./data_frame_turistveger.res', "rb") as fp:
    df_turistveger = pickle.load(fp)

update = False

names = df['Name'].unique()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

server = Flask(__name__)
app = Dash(server=server, external_stylesheets=external_stylesheets)

colorscales = px.colors.named_colorscales()

daterange = pd.date_range(start='2021', end='2024', freq='W')  # Todo
dff = df.iloc[::230, :]
fig = go.Figure()
fig.update_layout(margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
                  mapbox={
                      'accesstoken': access_token,
                      # 'center': {'lon': 139, 'lat': 36.5},
                      'style': 'outdoors',
                      # 'zoom': 4.5
                  },
                  # width=1600,
                  height=550,
                  showlegend=False)
data = go.Scattermapbox(
    lon=dff['Long'],
    lat=dff['Lat'],
    mode='markers',
    name='point',
    customdata=dff[['Name', 'Elevation', 'Distance', 'Duration']],
    hovertemplate=
    "<b>%{customdata[0]}</b><br>" +
    "<b>Elevation: %{customdata[1]}</b><br><br>" +
    "Distance: %{customdata[2]:,.2f}<br>" +
    "Duration: %{customdata[3]:.2f}<br>" +
    "<extra></extra>",
    marker=go.scattermapbox.Marker(color=list(dff['Elevation']),
                                   size=5,
                                   colorscale='rainbow',  # one of plotly colorscales
                                   showscale=True,
                                   # text=dff['Name']
                                   )
)
fig.add_trace(data)

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
                ['Selected', 'All'],
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
                ['Append', 'Remove Gap', 'Separate'],
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
                    ['Elevation', 'Speed', 'Distance', 'Duration', 'Temperature', 'Heart Rate', 'Power', 'Cadence',
                     'Accumulated Power', 'Calories', 'Pedal Smoothness', 'Torque Effectivenes', 'Name', 'Year',
                     'Type', 'Time', 'Lat', 'Long', 'Power Zone', 'Heart Rate Zone'],
                    value=['Elevation'],
                    id='crossfilter-yaxis-column',
                    multi=True
                )], style={'width': '99%', 'display': 'inline-block', 'float': 'right'}),
            html.Div([
                html.Div([dcc.Dropdown(
                    # df.columns,
                    ['Duration', 'Distance', 'Date'],
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
            config={'scrollZoom': True}
        )
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
    html.Div([
        dcc.Graph(id='x-time-series',
                  config={'scrollZoom': True}),
        dcc.Graph(id='y-time-series',
                  config={'scrollZoom': True}),
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
                # initial_visible_month=max(df['DateTime'])-timedelta(days=120),
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
                # style={'width': '50%', 'float': 'right'}
            )], style={'width': '22%', 'display': 'inline-block'}),
        html.Div([
            dcc.Upload(
                id='crossfilter-upload'
                , children=html.Div([
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
                options=["open-street-map", "carto-positron", "carto-darkmatter", "basic", "streets", "outdoors",
                         "light", "dark", "satellite", "satellite-streets",
                         "stamen-terrain", "stamen-toner", "stamen-watercolor"],
                value="outdoors",
                id='crossfilter-map-type',
            )], style={'width': '21%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                options=colorscales,
                value="rainbow",
                id='crossfilter-color-map',
            )], style={'width': '20%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                options=["point", "line", "heatmap", 'turistveger', 'polygon', 'tiles', 'all tiles', 'activity tiles'],
                value=["point"],
                id='crossfilter-map-plot',
                multi=True
            )], style={'width': '23%', 'display': 'inline-block'}),
        html.Div([
            dcc.Slider(1, 300, 25, value=1,
                       id='crossfilter-smoothing',
                       )], style={'width': '32%', 'display': 'inline-block'}),
    ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'}),
])


@callback(
    Output('crossfilter-indicator-scatter', 'figure', allow_duplicate=True),
    Input('crossfilter-activity-name', 'value'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-display-selection', 'value'),
    Input('crossfilter-virtual', 'value'),
    Input('crossfilter-activity-type', 'value'),
    State('crossfilter-append', 'value'),
    Input('crossfilter-shift', 'value'),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date'),
    State('crossfilter-smoothing', 'value'),
    State('crossfilter-map-plot', 'value'), prevent_initial_call=True)
def update_graph(activity_name, variable,
                 display, virtual, activity_type, append,
                 shift, start_date, end_date, smoothing, plot):
    global df
    patched_figure = Patch()

    dff = plotting_functions.get_plotting_dataset(df, activity_name, variable,
                                                  display, virtual, activity_type, append,
                                                  shift, start_date, end_date)
    for indx, dataset in enumerate(plot):
        if dataset == 'point':
            patched_figure['data'][indx]['lat'] = dff['Lat'].values
            patched_figure['data'][indx]['lon'] = dff['Long'].values
            patched_figure['data'][indx]['customdata'] = dff[['Name', variable[0], 'Distance', 'Duration']].values
            patched_figure['data'][indx]['marker'].update({'color': list(dff[variable[0]].rolling(smoothing).mean())})
    if dataset == 'line':
            patched_figure['data'][indx]['lat'] = dff['Lat'].values
            patched_figure['data'][indx]['lon'] = dff['Long'].values
            patched_figure['data'][indx]['customdata'] = dff[['Name', variable[0], 'Distance', 'Duration']].values
    if dataset == 'heatmap':
            patched_figure['data'][indx]['lat'] = dff['Lat'].values
            patched_figure['data'][indx]['lon'] = dff['Long'].values
    if dataset == 'Activity tiles':
            zoom = int(smoothing / 25 + 2)
            tiles = Tiles.check_tiles(zoom, dff)
            x = []
            y = []
            for tile in tiles:
                lon, lat = Tiles.tile_outline(tile, zoom)
                x.append(None)
                y.append(None)
                x.extend(lon)
                y.extend(lat)

            patched_figure['data'][indx]['lon'] = x
            patched_figure['data'][indx]['lat'] = y
    return patched_figure


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
    State('crossfilter-map-plot', 'value'),
    prevent_initial_call=True
)
def update_colorbar(colorbar, plot):
    print(f'Update colorscale {colorbar}')
    patched_figure = Patch()
    for indx, dataset in enumerate(plot):
        if dataset == 'point':
            print(f'{indx}')
            print(f"{patched_figure['data'][indx]['marker']['colorscale']}")
            patched_figure['data'][indx]['marker'].update({'colorscale': px.colors.get_colorscale(colorbar)})

            # patched_figure['data'][indx]['marker'].update({'colorscale':colorbar})
            print(f"{patched_figure['data'][indx]['marker']['colorscale']}")
            print('Found Colorbar')
    return patched_figure



@app.callback(
    Output('crossfilter-indicator-scatter', 'figure', allow_duplicate=True),
    State('crossfilter-map-plot', 'value'),
    State('crossfilter-activity-name', 'value'),
    State('crossfilter-yaxis-column', 'value'),
    State('crossfilter-display-selection', 'value'),
    State('crossfilter-virtual', 'value'),
    State('crossfilter-activity-type', 'value'),
    Input('crossfilter-append', 'value'),
    State('crossfilter-shift', 'value'),
    State('my-date-picker-range', 'start_date'),
    State('my-date-picker-range', 'end_date'),
    State('crossfilter-smoothing', 'value'),

    prevent_initial_call=True
)
def update_append(plot, activity_name, variable,
                             display, virtual, activity_type, append,
                             shift, start_date, end_date, smoothing):
    global df
    print(f'{variable = }')
    if variable.__len__()>0:
        if variable[0] == 'Distance'or variable[0] == 'Duration':
            if plot.__contains__('point'):
                dff = plotting_functions.get_plotting_dataset(df, activity_name, variable,
                                                              display, virtual, activity_type, append,
                                                              shift, start_date, end_date)
                patched_figure = Patch()
                for indx, dataset in enumerate(plot):
                    if dataset == 'point':
                        patched_figure['data'][indx]['customdata'] = dff[['Name', variable[0], 'Distance', 'Duration']].values
                        patched_figure['data'][indx]['marker'].update({'color': list(dff[variable[0]].rolling(smoothing).mean())})
                        return patched_figure
    return Patch()


@app.callback(
    Output('crossfilter-indicator-scatter', 'figure', allow_duplicate=True),
    Input('crossfilter-map-plot', 'value'),
    State('crossfilter-activity-name', 'value'),
    State('crossfilter-yaxis-column', 'value'),
    State('crossfilter-display-selection', 'value'),
    State('crossfilter-virtual', 'value'),
    State('crossfilter-activity-type', 'value'),
    State('crossfilter-append', 'value'),
    State('crossfilter-shift', 'value'),
    State('my-date-picker-range', 'start_date'),
    State('my-date-picker-range', 'end_date'),
    Input('crossfilter-smoothing', 'value'),
    State('crossfilter-map-type', 'value'),
    State('crossfilter-color-map', 'value'),

    prevent_initial_call=True
)
def update_Map_plotting_type(plot, activity_name, variable,
                             display, virtual, activity_type, append,
                             shift, start_date, end_date, smoothing, map_type, color_map):
    global df
    dff = plotting_functions.get_plotting_dataset(df, activity_name, variable,
                                                  display, virtual, activity_type, append,
                                                  shift, start_date, end_date)

    if plot.__len__() == 0:
        print('Empty')
        return Patch()
    data = []
    for idx, value in enumerate(plot):
        show_legend = (idx == 0)
        print(f'{value = }')
        if value == 'point':
            print('point')
            if variable.__len__()>0:
                output_variable = variable[0]
                output = go.Scattermapbox(
                    lon=dff['Long'],
                    lat=dff['Lat'],
                    mode='markers',
                    name='point',
                    customdata=dff[['Name', output_variable, 'Distance', 'Duration']],
                    hovertemplate=
                    "<b>%{customdata[0]}</b><br>" +
                    "<b>" + output_variable + ": %{customdata[1]}</b><br><br>" +
                    "Distance: %{customdata[2]:,.2f} km<br>" +
                    "Duration: %{customdata[3]:.2f} min<br>" +
                    "<extra></extra>",
                    marker=go.scattermapbox.Marker(color=list(dff[output_variable].rolling(smoothing).mean()),
                                                   size=5,
                                                   colorscale=color_map,  # one of plotly colorscales
                                                   showscale=show_legend
                                                   ))
                #print(output)
                data.append(output)
                print('Added Point')
        if value == 'line':
            print('line')
            dff.loc[dff["Distance"] < 0.1 * max(1, 50 / 5), "Lat"] = np.NAN
            if variable.__len__()>0:
                output_variable = variable[0]
                #for name in dff['Name'].unique(): # To be used to ges single lines per activity
                #    dfn = dff[dff['Name'] ==  name]
                data.append(go.Scattermapbox(
                    lat=dff['Lat'],
                    lon=dff['Long'],
                    mode='lines',
                    customdata=dff[['Name', output_variable, 'Distance', 'Duration']],
                    hovertemplate=
                    "<b>%{customdata[0]}</b><br>" +
                    "<b>" + output_variable + ": %{customdata[1]}</b><br><br>" +
                    "Distance: %{customdata[2]:,.2f} km<br>" +
                    "Duration: %{customdata[3]:.2f} min<br>" +
                    "<extra></extra>",
                    name='Lines', ))

        if value == 'heatmap':
            print('heatmap')
            heatmap = go.Densitymapbox(lat=dff['Lat'], lon=dff['Long'],  # z=dff['Speed'],
                                       radius=int(smoothing / 10) + 1)
            data.append(heatmap)
        if value == 'turistveger':
            turistveger = go.Scattermapbox(
                lon=df_turistveger['Long'],
                lat=df_turistveger['Lat'],
                mode='markers',
                name='Trace',
                customdata=dff[['Name']],
                hovertemplate=
                "<b>%{customdata[0]}</b><br>" +
                "<extra></extra>",
                marker=go.scattermapbox.Marker(color=list(df_turistveger['Elevation']),
                                               size=5,
                                               colorscale='rainbow',  # one of plotly colorscales
                                               showscale=show_legend,
                                               ))
            data.append(turistveger)
        if value == 'polygon':
            with open("./polygons.res", "rb") as fp:
                polygons_list = pickle.load(fp)
            with open("./activity_polygons.res", "rb") as fp:
                activity_polygons = pickle.load(fp)
            polygons_list.reverse()
            gdf = gpd.GeoSeries(polygons_list)
            gdf.crs = "epsg:4326"
            areas = gdf.to_crs({'init': 'epsg:32633'}) \
                .map(lambda p: p.area / 10 ** 6)
            gdf = gpd.GeoDataFrame(gdf)
            gdf = gdf.assign(area=areas)
            gdf = gdf.set_geometry(0)

            polygons_plot = go.Choroplethmapbox(geojson=json.loads(gdf.to_json()),
                                                locations=gdf.index, z=gdf['area'],
                                                colorscale="Viridis",
                                                marker_opacity=0.6,
                                                hovertemplate=
                                                "<b>%{z:.2f} km^2</b><br>" +
                                                "<extra></extra>",
                                                showscale=show_legend)
            data.append(polygons_plot)

        # Tile plots
        zoom = int(smoothing / 25 + 2)
        if value == 'All tiles':
            print(f'{zoom = }')
            long, lat = Tiles.generate_reduced_lines(zoom)
            data.append(go.Scattermapbox(
                lon=long,
                lat=lat,
                mode='lines',
                fillcolor='red',
                line=dict(color='red', width=0.5),
                name='All-Lines',
                opacity=.25,
                hoverinfo=None,
                hoverlabel=None
            ))

        if value == 'tiles':
            with open('Tiles.res', "rb") as fp:
                all_Tiles = pickle.load(fp)
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
                fillcolor='green',
                line=dict(color='green'),
                name='all',
                hoverinfo=None,
                hoverlabel=None
            ))
        if value == 'Activity tiles':
            tiles = Tiles.check_tiles(zoom, dff)
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
                name='all',
                hoverinfo=None,
                hoverlabel=None
            ))

    print(data.__len__())

    fig = Patch()
    fig['data'] = data
    return fig


@callback(
    Output('x-time-series', 'figure'),
    Input('crossfilter-activity-name', 'value'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-yaxis-type', 'value'),
    Input('crossfilter-append', 'value'),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date'),
    Input('crossfilter-smoothing', 'value'), )
def update_x_timeseries(activity_name, yaxis_column_name, axis_type, append, start_date, end_date, smoothing):
    global df
    return plotting_functions.update_x_timeseries(df, activity_name, yaxis_column_name, axis_type, append, start_date,
                                                  end_date, smoothing)


@callback(
    Output('y-time-series', 'figure'),
    Input('crossfilter-activity-name', 'value'),
    Input('crossfilter-display-selection', 'value'),
    Input('crossfilter-overview-plot', 'value'),
    Input('crossfilter-yaxis-type', 'value'),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date'),
    Input('crossfilter-append', 'value'),
    Input('crossfilter-virtual', 'value'),
    Input('crossfilter-activity-type', 'value'))
def update_overview_plots(activity_name, display, plot, axis_type, start_date, end_date, append, virtual,
                          activity_type):
    global df
    return plotting_functions.update_overview_plots(df, activity_name, display, plot, axis_type, start_date, end_date,
                                                    append, virtual, activity_type)


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=False, port=8055)
