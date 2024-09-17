import math
import matplotlib.pyplot as plt
import numpy as np
from main import *

def coord_from_tile(x, y, zoom=14):
    n = 2 ** zoom
    if y is None:
        s = x.split('_')
        x = int(s[0])
        y = int(s[1])
    lat = math.atan(math.sinh(math.pi * (1 - 2 * y / n))) * 180.0 / math.pi
    lon = x / n * 360.0 - 180.0
    return lat, lon

def geom_from_tile(x, zoom):
    s = x.split('_')
    x = int(s[0])
    y = int(s[1])
    return [list(coord_from_tile(x, y, zoom))[::-1], list(coord_from_tile(x + 1, y + 1, zoom))[::-1]]


def tile_outline(x, zoom):
    coords = geom_from_tile(x, zoom)
    return [coords[0][0], coords[1][0], coords[1][0], coords[0][0], coords[0][0]],[coords[0][1], coords[0][1], coords[1][1], coords[1][1], coords[0][1]]

def get_lattitude(i,n):
    return math.atan(math.sinh(math.pi * (1 - 2 * i / n))) * 180.0 / math.pi

def generate_lines(zoom=14):
    n = 2**zoom
    lat = []
    long = []
    lat_index = 0
    #Longitudinal lines
    while get_lattitude(lat_index,n) > -85:
        lat.extend([get_lattitude(lat_index,n)]*n)
        long.extend(np.linspace(-180, 180, n+1)[:-1])
        lat.append(None)
        long.append(None)
        lat_index = lat_index + 1
    for longitude in np.linspace(-180, 180, n+1)[:-1]:
        for lattitude in lat[::n+1][:n]:
            lat.append(lattitude)
            long.append(longitude)
        lat.append(None)
        long.append(None)
    return long, lat


def generate_reduced_lines(zoom=14):
    n = 2**zoom
    lat_index = 0
    latitude = get_lattitude(lat_index,n)
    lat = [latitude,latitude,None]
    latitudes = [latitude]
    long = [-180, 180, None]
    #Longitudinal lines
    while latitude > -85:
        lat_index = lat_index + 1
        latitude = get_lattitude(lat_index, n)
        latitudes.append(latitude)
        lat.extend([latitude]*2)
        long.extend([-180, 180])
        lat.append(None)
        long.append(None)
    for longitude in np.linspace(-180, 180, n+1)[:-1]:
        for latitude in [min(latitudes), max(latitudes)]:
            lat.append(latitude)
            long.append(longitude)
        lat.append(None)
        long.append(None)
    return long, lat

def check_tiles(zoom, df):
    df = df[['Lat', 'Long']]
    tiles = []
    n = 2 ** zoom
    lat_index = 0
    lat = get_lattitude(lat_index,n)
    lat_next = get_lattitude(lat_index+1,n)
    longitudes = np.linspace(-180, 180, n + 1)
    max_lat = df['Lat'].max()
    while lat > -85:
        if max_lat > lat_next:
            df = df[df['Lat'] < lat]
            lat_mask = df['Lat'] > lat_next
            if lat_mask.values.__contains__(True):
                print(f'Found Tile at Lattidude {lat}')
                dfn = df[['Long']]
                dfn = dfn[lat_mask]
                min_long = dfn['Long'].min()
                for idx, longitude in enumerate(longitudes[:-1]):
                    if longitudes[idx+1] > min_long:
                        mask = dfn['Long'] > longitude
                        if not mask.values.__contains__(True):
                            break
                        dfn = dfn[mask]
                        min_long = dfn['Long'].min()
                        if (dfn['Long'] < longitudes[idx+1]).values.__contains__(True):
                            uid = "{0}_{1}".format(idx, lat_index)
                            tiles.append(uid)
                            #print(f'Found Tile {uid}, Lat {lat}, Long {longitude}')
            if df.shape[0] == 0:
                break
            max_lat = df['Lat'].max()
        lat_index = lat_index+1
        lat = get_lattitude(lat_index, n)
        lat_next = get_lattitude(lat_index + 1, n)
    return tiles


def check_tiles_v2(zoom, df):
    df = df[df['Type'] != 'Virtual Ride']
    df = df[['Lat', 'Long']]
    tiles = []
    n = 2 ** zoom
    lat_index = 0
    lat = get_lattitude(lat_index,n)
    lat_next = get_lattitude(lat_index+1,n)
    longitudes = np.linspace(-180, 180, n + 1)
    max_lat = df['Lat'].max()
    while lat > -85:
        if max_lat > lat_next:
            df = df[df['Lat'] < lat]
            lat_mask = df['Lat'] > lat_next
            if lat_mask.values.__contains__(True):
                print(f'Found Tile at Lattidude {lat}')
                dfn = dfn[lat_mask]
                min_long = dfn['Long'].min()
                for idx, longitude in enumerate(longitudes[:-1]):
                    if longitudes[idx+1] > min_long:
                        mask = dfn['Long'] > longitude
                        if not mask.values.__contains__(True):
                            break
                        dfn = dfn[mask]
                        min_long = dfn['Long'].min()
                        if (dfn['Long'] < longitudes[idx+1]).values.__contains__(True):
                            uid = "{0}_{1}".format(idx, lat_index)
                            tiles.append(uid)
                            #print(f'Found Tile {uid}, Lat {lat}, Long {longitude}')
            if df.shape[0] == 0:
                break
            max_lat = df['Lat'].max()
        lat_index = lat_index+1
        lat = get_lattitude(lat_index, n)
        lat_next = get_lattitude(lat_index + 1, n)
    return tiles

def analyse_dataframe(df):
    all_Tiles = dict()
    df = df[df['Type'] != 'Virtual Ride']
    df = df[['Lat', 'Long']]
    for level in [2,3,4,5,6,7,8,9,10,11,12,13,14, 15, 16,17]:
        print(f'{level = }')
        tiles = check_tiles(level, df)
        all_Tiles.update({level:tiles})

    with open("Tiles.res", "wb") as fp:  # Pickling
        pickle.dump(all_Tiles, fp)

if __name__ == '__main__':
    x,y = generate_reduced_lines(7)
    plt.plot(x,y)
    plt.show()
    with open('./data_frame.res', "rb") as fp:
        df = pickle.load(fp)
    import timeit

    timeit.timeit("tiles = check_tiles_v2(12, df)", number=3, globals=globals())
    timeit.timeit("tiles = check_tiles(12, df)", number=3, globals=globals())
    tiles = check_tiles(12, df)
    x=[]
    y=[]
    for tile in tiles:
        lon, lat = tile_outline(tile, 12)
        x.append(None)
        y.append(None)
        x.extend(lon)
        y.extend(lat)
    plt.plot(x,y)
    plt.show()

    analyse_dataframe(df)
    print('Done')