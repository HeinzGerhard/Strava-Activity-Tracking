import math

import matplotlib.pyplot as plt
import numpy as np

import main
from scipy import spatial
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union
import geopandas as gpd
import pickle



start_points = []
end_points = []

def get_distance(node_1,node_2):
    x = node_1[0]-node_2[0]
    y = node_1[1]-node_2[1]
    dist = math.sqrt(x**2+y**2)
    return dist


def make_line_string(activity, threshold=5):
    nodes = []

    nodes.append(activity.lon)
    nodes.append(activity.lat)
    nodes = np.transpose(nodes)
    nodes = nodes[~np.isnan(nodes).any(axis=1)]
    mask = np.all(np.equal(nodes, None), axis=1)
    nodes = nodes[~mask]

    if not activity.virtual and nodes.__len__() > 0:
        edited_start = False
        edited_end = False
        tolerance = 170/(1852*60)
        exceptions = ['TURNo1-Day3', 'Helinki-Part 3']
        for exception in exceptions:
            if activity.name.__contains__(exception):
                tolerance = 2700/(1852*60)
        for start_point in start_points:
            distance = get_distance(nodes[0], start_point)
            if distance < tolerance:
                nodes = np.vstack([start_point, nodes])
                edited_start = True
                break
        if not edited_start:
            start_points.append(nodes[0])
        for start_point in start_points:
            distance = get_distance(nodes[-1], start_point)
            if distance < tolerance:
                nodes = np.vstack([nodes, start_point])
                edited_end = True
                break
        if not edited_end:
            start_points.append(nodes[-1])
    line = LineString(nodes)
    return line

def area_intersection(line_1, line_2):
    return line_1.minimum_rotated_rectangle.intersects(line_2.minimum_rotated_rectangle)
    bounds_line_1 = line_1.bounds
    bounds_line_2 = line_2.bounds
    if bounds_line_1[2]>bounds_line_2[0] and bounds_line_1[0]<bounds_line_2[2]:
        if bounds_line_1[3]>bounds_line_2[1] and bounds_line_1[3]>bounds_line_2[1]:
            return True
    return False

def analyse_polygons(lines, name):
    all_polygons = []
    for line in lines:
        try:
            all_lines = unary_union(line)
            polygons = polygonize(all_lines)
            polygons = list(polygons)
            for polygon in polygons:
                #print(f'{polygon.area = }')
                if polygon.area > 1e-6:
                    all_polygons.append(Polygon(polygon.exterior.coords))
        except Exception as e:
            print(e)
    return all_polygons


def analyse_lines(lines):
    polygons = polygonize(unary_union(lines))
    return_polygons = []
    for polygon in polygons:
        if polygon.area > 0.5e-6:
            return_polygons.append(Polygon(polygon.exterior.coords))
    return return_polygons


def analyse_activities():

    activities = main.load_data_file()
    for activity in activities:
        print(f'{activity.name}')
        activity.intersections = set()
        #try:
        activity.line = make_line_string(activity)
        activity.polygons = analyse_polygons([activity.line], activity.name)
    polygons = []
    activity_polygons = dict()
    for activity in activities:
        if hasattr(activity, 'polygons'):
            activity_polygons.update({activity.name : activity.polygons})
            for act_polygon in activity.polygons:
                polygons.append(act_polygon)

    ## Calculate Intersections

    for idx, activity in enumerate(activities):
        #activity = activities[idx]
        if not activity.virtual and not activity.commute:
            #print(f'Intersections {activity.name}')
            if activity.line != []:
                for idx_2,activity_2 in enumerate(activities[idx+1:]):
                    #activity_2 = activities[idx_2+idx+1]
                    if activity_2.line != [] and not activity_2.virtual \
                            and not activity_2.commute:
                        try:
                            if  not activity.intersections.__contains__(idx_2+idx+1):
                                if activity.line.intersects(activity_2.line):
                                    activity.intersections.add(idx_2+idx+1)
                        except:
                            print(f'{activity.name}, {activity_2.name}')
                for idx in activity.intersections:
                    activities[idx].intersections.update(activity.intersections)
            print(f'Intersections {activity.name}:\t{activity.intersections.__len__()}')

    sets = []

    for idx, activity in enumerate(activities):
        intersections = set(activity.intersections)
        if intersections.__len__() >0:
            intersections.add(idx)
            new = True
            for existing_set in sets:
                if intersections.issubset(existing_set):
                    new = False
                    break
                if not intersections.isdisjoint(existing_set):
                    existing_set.update(intersections)
                    new = False
                    break
            if new:
                sets.append(intersections)
    final_set = sets


    for intersect in final_set:
        print(f'Working on Set lenght {intersect.__len__()}')
        lines = []
        for idx in intersect:
            lines.append(activities[idx].line)
        return_polygons = (analyse_lines(lines))
        for polygon in return_polygons:
            polygons.append(polygon)

    print(f'Working on finalisation')

    final_polygons = unary_union(polygons)
    temp_polygons_list = list(final_polygons.geoms)
    polygons_list = []
    for polygon in temp_polygons_list:
        if polygon.area > 2.5e-6:
            return_polygons.append(Polygon(polygon.exterior.coords))
            polygons_list.append(Polygon(polygon.exterior.coords))

    with open("./polygons.res", "wb") as fp:  # Pickling
        pickle.dump(polygons_list, fp)
    with open("./activity_polygons.res", "wb") as fp:  # Pickling
        pickle.dump(activity_polygons, fp)

    gdf = gpd.GeoSeries(polygons_list)
    gdf.crs = "epsg:4326"
    areas = gdf.to_crs({'init': 'epsg:32633'})\
                   .map(lambda p: p.area / 10**6)
    print(f'Total ridden area: {sum(areas)} km^2')
    print(f'Area of the earth: {sum(areas)/148326000*100} %')

if __name__ == '__main__':
    analyse_activities()
