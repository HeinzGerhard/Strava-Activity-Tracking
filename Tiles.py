import math
from traceback import print_tb

import matplotlib.pyplot as plt
import numpy as np
from numpy.ma.core import array
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray

from main import *
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union
import geopandas as gpd
import time as timeit

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
    start = timeit.time()
    df = df[['Lat', 'Long', 'DS']]
    tiles = []
    area = []
    visits = []
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
                #print(f'Found Tile at Lattidude {lat}')
                dfn = df[['Long', 'DS']]
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
                            dfs = dfn[dfn['Long'] < longitudes[idx+1]]
                            area.append(dfs['DS'].sum())
                            #print(f'Found Tile {uid}, Lat {lat}, Long {longitude}')
            if df.shape[0] == 0:
                break
            max_lat = df['Lat'].max()
        lat_index = lat_index+1
        lat = get_lattitude(lat_index, n)
        lat_next = get_lattitude(lat_index + 1, n)
    end = timeit.time()
    print(f'{end-start} seconds')
    return tiles, area

def split_dataframe(values, parameter, zoom, df, reverse):
    n = 2 ** zoom
    dfs = [[df, 0]]
    numbers = n
    for i in range(zoom):
        split_index = int(numbers//2)
        dfs_new=[]
        for df, offset in dfs:
            #print(f'Latitude index {offset+split_index}, {latitudes.__len__()}')
            if offset+split_index < values.__len__():
                lat = values[offset+split_index]
                lat_mask = df[parameter] > lat

                if reverse:
                    lat_mask = ~lat_mask
                #print(f'Latitude {lat}')
                if lat_mask.values.__contains__(True):
                    dfn = df[lat_mask]
                    dfs_new.append([dfn, offset])
                if lat_mask.values.__contains__(False):
                    dfn = df[~lat_mask]
                    dfs_new.append([dfn, offset + split_index])
                else:
                    pass
            else:
                dfs_new.append([df, offset])
        dfs = dfs_new
        numbers=numbers/2
    return dfs

def check_tiles_v3(zoom, df):
    start = timeit.time()
    df = df[['Lat', 'Long', 'DS']]
    tiles = []
    area = []
    n = 2 ** zoom
    lat_index = 0
    lat = get_lattitude(lat_index,n)
    latitudes = [lat]
    while lat > -85:
        lat_index = lat_index + 1
        lat = get_lattitude(lat_index, n)
        latitudes.append(lat)
    longitudes = np.linspace(-180, 180, n + 1)
    #print(f'{latitudes = }')

    dfs = split_dataframe(latitudes, 'Lat', zoom, df, False)
    #print('Finished Split')

    for df, lat_index  in dfs:
        dfn = df[['Long', 'DS']]
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
                    dfs = dfn[dfn['Long'] < longitudes[idx+1]]
                    area.append(dfs['DS'].sum())
    end = timeit.time()
    print(f'V3: {end-start} seconds or {tiles.__len__()/(end - start)} Tiles per second')
    return tiles, area


def check_tiles_v4(zoom, df):
    start = timeit.time()
    df = df[['Lat', 'Long', 'DS']]
    df = df.dropna()
    tiles = []
    area = []
    n = 2 ** zoom
    lat_index = 0
    lat = get_lattitude(lat_index, n)
    latitudes = [lat]
    while lat > -85:
        lat_index = lat_index + 1
        lat = get_lattitude(lat_index, n)
        latitudes.append(lat)
    longitudes = np.linspace(-180, 180, n + 1)
    # print(f'{latitudes = }')
    df['Long Bins'] = pd.cut(df['Long'], longitudes)
    dfs = split_dataframe(latitudes, 'Lat', zoom, df, False)
    # print('Finished Split')

    for df, lat_index in dfs:
        dfn = df[['Long Bins', 'Long', 'DS']]
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
                    dfs = dfn[dfn['Long'] < longitudes[idx+1]]
                    area.append(dfs['DS'].sum())

    end = timeit.time()
    print(f'V4: {end - start} seconds or {tiles.__len__()/(end - start)} Tiles per second')
    return tiles, area


def compute_cluster(tiles):
    def is_cluster(x, y):
        for d_tile in [[-1,0], [1,0], [0,-1],[0,1]]:
            uid = "{}_{}".format(x+d_tile[0], y+d_tile[1])
            if uid not in tiles :
                return False
        return True
    start = timeit.time()

    values = []
    for tile in tiles:
        #print(tile)
        x = int(tile.split('_')[0])
        y = int(tile.split('_')[1])
        if is_cluster(x, y):
            values.append(1)
        else:
            values.append(0)
    end = timeit.time()
    print(f'CLuster: {end - start} seconds')

    return values


def compute_cluster_v2(tiles, tiles_array):
    def is_cluster(x, y):
        for d_tile in [[-1,0], [1,0], [0,-1],[0,1]]:
            try:
                if not tiles_array[x+d_tile[0], y+d_tile[1]]:
                    return False
            except: return False
        return True

    start = timeit.time()
    values = []
    for tile in tiles:
        #print(tile)
        x = int(tile.split('_')[0])
        y = int(tile.split('_')[1])
        if is_cluster(x, y):
            values.append(1)
        else:
            values.append(0)
    end = timeit.time()
    print(f'Cluster_v2: {end - start} seconds')

    return values


def compute_cluster_size(tiles, clusters):
    def get_connections(tile):
        uuids = []
        x = int(tile.split('_')[0])
        y = int(tile.split('_')[1])
        for d_tile in [[-1,0], [1,0], [0,-1],[0,1]]:
                uid = "{}_{}".format(x+d_tile[0], y+d_tile[1])
                uuids.append(uid)
        return uuids
    cluster_sizes = np.zeros(len(clusters))
    max_cluster_size = 0
    max_cluster = set()
    for idx, cluster in enumerate(clusters):
        if cluster  == 1 and cluster_sizes[idx] == 0:
            cluster_size = 1
            cluster_Set = set()
            analysed_Set = set()
            cluster_Set.add(tiles[idx])
            while cluster_Set > analysed_Set:
                for tile in sorted(cluster_Set-analysed_Set):
                    connections = get_connections(tile)
                    for connection in connections:
                        if not cluster_Set.__contains__(connection):
                            indx = tiles.index(connection)
                            if clusters[indx] == 1:
                                cluster_Set.add(connection)
                                cluster_size = cluster_size + 1
                    analysed_Set.add(tile)
            if cluster_size > max_cluster_size:
                max_cluster_size = cluster_size
                max_cluster = cluster_Set
            for tile in sorted(cluster_Set):
                indx = tiles.index(tile)
                cluster_sizes[indx] = cluster_size
    print(f'{max_cluster_size = }')
    return cluster_sizes


def compute_max_square(tiles, level):

    def is_square(x, y, m):
        for dx in range(m).__reversed__():
            for dy in range(m).__reversed__():
                uid = "{}_{}".format(x+dx, y+dy)
                if uid not in tiles : return False
        return True

    start = timeit.time()
    max_square = 0
    x_max = 0
    y_max = 0
    for tile in tiles:
        #print(tile)
        x = int(tile.split('_')[0])
        y = int(tile.split('_')[1])
        while is_square(x, y, max_square+1):
            max_square += 1
            x_max = x
            y_max = y
    polygons_list = []
    for x in range(x_max, x_max+max_square):
        for y in range(y_max, y_max+max_square):
            lon, lat = tile_outline("{}_{}".format(x,y), level)
            nodes = []
            nodes.append(lon)
            nodes.append(lat)
            nodes = np.transpose(nodes)
            line = LineString(nodes)
            polygon = polygonize(line)
            polygons_list.append(polygon[0])

    gdf = gpd.GeoSeries(polygons_list)
    gdf.crs = "epsg:4326"
    gdf = gpd.GeoDataFrame(gdf)
    gdf = gdf.assign(area=max_square)
    gdf = gdf.set_geometry(0)
    end = timeit.time()
    print(f'Max_square: {end - start} seconds')
    return gdf


def compute_max_square_v2(tiles, tile_array, level):

    def is_square(x, y, m):
        for dx in range(m).__reversed__():
            for dy in range(m).__reversed__():
                try:
                    if not tile_array[x+dx, y+dy] : return False
                except: return False
        return True

    start = timeit.time()
    max_square = 0
    x_max = 0
    y_max = 0
    for tile in tiles:
        x = int(tile.split('_')[0])
        y = int(tile.split('_')[1])
        while is_square(x, y, max_square+1):
            max_square += 1
            x_max = x
            y_max = y
    polygons_list = []
    for x in range(x_max, x_max+max_square):
        for y in range(y_max, y_max+max_square):
            lon, lat = tile_outline("{}_{}".format(x,y), level)
            nodes = []
            nodes.append(lon)
            nodes.append(lat)
            nodes = np.transpose(nodes)
            line = LineString(nodes)
            polygon = polygonize(line)
            polygons_list.append(polygon[0])

    gdf = gpd.GeoSeries(polygons_list)
    gdf.crs = "epsg:4326"
    gdf = gpd.GeoDataFrame(gdf)
    gdf = gdf.assign(area=max_square)
    gdf = gdf.set_geometry(0)
    end = timeit.time()
    print(f'Max_square_v2: {end - start} seconds')
    return gdf


def check_tiles_fast(zoom, df):
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


def analyse_performance(df):
    df = df[df['Type'] != 'Virtual Ride']
    df = df[['Lat', 'Long', 'DS']]
    for level in [2,3,4,5,6,7,8,9,10,11,12,13,14, 15, 16,17]:
        print(f'{level = }')
        tiles, area = check_tiles_v3(level, df)
        tiles_2, area_2 = check_tiles_v4(level, df)
        tiles_set = set(tiles)
        tiles_2_set = set(tiles_2)
        if tiles_set == tiles_2_set:
            if area == area_2 or True:
                print('Good Results')
            else:
                print('Bad Results')
                print(f'{area = }')
                print(f' vs {area_2 = }')
        else:
            print('Bad Results')
            if tiles_set > tiles_2_set:
                print(f'Missing: {tiles_set - tiles_2_set}')
            else:
                print(f'Addistional: {tiles_2_set - tiles_set}')
                print(tiles_set)
                print(tiles_2_set)

def tyles_to_array(tiles, zoom):
    start = timeit.time()
    n = 2 ** zoom
    tile_array = np.full((n, n), False, dtype=bool)
    for tile in tiles:
        x = int(tile.split('_')[0])
        y = int(tile.split('_')[1])
        tile_array[x, y] = True
    end = timeit.time()
    print(f'Tiles to array: {end - start} seconds')
    return tile_array

def analyse_dataframe(df):
    all_Tiles = dict()
    all_Areas = dict()
    all_gdfs = dict()
    all_Max_Squares = dict()
    all_clusters = dict()
    all_cluster_sizes = dict()
    df = df[df['Type'] != 'Virtual Ride']
    df = df[['Lat', 'Long', 'DS']]
    for level in [2,3,4,5,6,7,8,9,10,11,12,13,14, 15, 16,17]:
        print(f'{level = }')
        tiles, area = check_tiles_v3(level, df)
        all_Tiles.update({level:tiles})
        all_Areas.update({level:area})
        polygons_list = []
        for tile in tiles:
            lon, lat = tile_outline(tile, level)
            nodes = []
            nodes.append(lon)
            nodes.append(lat)
            nodes = np.transpose(nodes)
            line = LineString(nodes)
            polygon = polygonize(line)
            polygons_list.append(polygon[0])

        gdf = gpd.GeoSeries(polygons_list)
        gdf.crs = "epsg:4326"
        gdf = gpd.GeoDataFrame(gdf)
        gdf = gdf.assign(area=area)
        gdf = gdf.set_geometry(0)
        all_gdfs.update({level:gdf})
        print(f'Found {tiles.__len__()} Tiles')
        tile_array = tyles_to_array(tiles, level)
        print('Max_Square')
        all_Max_Squares.update({level:compute_max_square_v2(tiles, tile_array, level)})
        print('Clusters')
        #clusters = compute_cluster(tiles)
        clusters = compute_cluster_v2(tiles, tile_array)
        all_clusters.update({level:clusters})
        print('Cluster_Sizes')
        all_cluster_sizes.update({level:compute_cluster_size(tiles, clusters)})

        with open("Tiles.res", "wb") as fp:  # Pickling
            pickle.dump([all_Tiles, all_Areas, all_gdfs, all_Max_Squares, all_clusters, all_cluster_sizes], fp)

if __name__ == '__main__':
    with open('./data_frame.res', "rb") as fp:
        df = pickle.load(fp)
    analyse_performance(df)
    x,y = generate_reduced_lines(7)
    plt.plot(x,y)
    plt.show()
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