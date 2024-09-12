import math

import matplotlib.pyplot as plt
import numpy as np

from main import *
from scipy import spatial
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union
import geopandas as gpd



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
    #for idx,node_1 in enumerate(nodes):
    #    for idx_2, node_2 in enumerate(nodes[idx+1:]):
    #        if get_distance(node_1, node_2) < threshold / (1852 * 60):
    #            nodes[idx+idx_2+1] = node_1
                #print(f'replaced {idx = } and {idx_2 =}')


    #distance = get_distance(nodes[0], nodes[-1])
    #print(f'{distance = } {distance*(1852*60) = }')
    #closed = distance < 170/(1852*60)
    #if distance < 170/(1852*60):
    #    nodes = np.vstack([nodes[:-2], nodes[0],nodes[-1]])
    #    print(f'Closed Loop')
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
                print(f'\tFixed Startpoint')
                edited_start = True
                break
        if not edited_start:
            start_points.append(nodes[0])
        for start_point in start_points:
            distance = get_distance(nodes[-1], start_point)
            if distance < tolerance:
                nodes = np.vstack([nodes, start_point])
                print(f'\tFixed Endpoint')
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
    #gs_l = gpd.GeoSeries(lines)
    #ax = gs_l.plot()
    #plt.title(f'All Lines')
    #plt.show()
    polygons = polygonize(unary_union(lines))
    return_polygons = []
    for polygon in polygons:
        if polygon.area > 0.5e-6:
            return_polygons.append(Polygon(polygon.exterior.coords))
            #print(f'{polygon.area = }')
    return return_polygons


def analyse_activities():

    start = timeit.time()
    activities = load_data_file()
    #activities = activities[580:595]
    loaded = timeit.time()
    print(f'reading: {loaded - start}')
    for activity in activities:
        print(f'{activity.name}')
        activity.intersections = set()
        #try:
        activity.line = make_line_string(activity)
        activity.polygons = analyse_polygons([activity.line], activity.name)
        #except Exception as e:
        #    print(f'Error: {activity.name}')
        #    print(e)

    linestring = timeit.time()

    print(f'reading: {loaded - start}, Line_string: {linestring - loaded},')
    polygons = []
    activity_polygons = dict()
    for activity in activities:
        if hasattr(activity, 'polygons'):
            activity_polygons.update({activity.name : activity.polygons})
            for act_polygon in activity.polygons:
                polygons.append(act_polygon)

            #new = True
            #for polygon in polygons:
            #    if not act_polygon.disjoint(polygon):
            #        new = False
            #if new:
            #    polygons.append(act_polygon)

    collect_polygons = timeit.time()

    print(f'reading: {loaded - start}, Line_string: {linestring - loaded},'
          f' collect polygons: {collect_polygons - linestring}')

    ## Calculate Intersections

    for idx, activity in enumerate(activities):
        if not activity.virtual and not activity.commute:
            #print(f'Intersections {activity.name}')
            if activity.line != []:
                for idx_2,activity_2 in enumerate(activities[idx+1:]):
                    if activity.line != [] and not activity_2.virtual \
                            and not activity_2.commute:
                        try:
                            if activity.intersections.__contains__(idx_2+idx+1):
                                activity_2.intersections.update(activity.intersections)
                            elif activity.line.intersects(activity_2.line):
                                activity.intersections.add(idx_2+idx+1)
                                activity_2.intersections.update(activity.intersections)
                        except:
                            print(f'{activity.name}, {activity_2.name}')
            print(f'Intersections {activity.name}\t{activity.intersections}')


    intersections_time = timeit.time()
    print(f'reading: {loaded - start}, Line_string: {linestring - loaded},'
          f' collect polygons: {collect_polygons - linestring}, intersections: {intersections_time - collect_polygons}')
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
    if False:
        change = True
        while change:
            change = False
            for idx, set_1 in enumerate(sets):
                for idx_2, set_2 in enumerate(sets[idx+1:]):
                    if not set(set_1).isdisjoint(set_2) and not set(set_1).issuperset(set_2):
                        set_1 = set_1.union(set_2)
                        #set_1 = set(set_1)
                        change = True
        final_set = []

        for set_1 in sets:
            use = True
            for set_2 in sets:
                if set(set_1) < set(set_2):
                    use = False
                    break
            if use:
                final_set.append(set(set_1))


    sets_time = timeit.time()

    print(f'reading: {loaded - start}, Line_string: {linestring - loaded},'
          f' collect polygons: {collect_polygons - linestring}, Intersectiosn: {intersections_time - collect_polygons},'
          f'Sets: {sets_time - intersections_time}')

    for intersect in final_set:
        print(f'Working on Set\n{intersect}\n{intersect.__len__() = }')
        lines = []
        for idx in intersect:
            lines.append(activities[idx].line)
        return_polygons = (analyse_lines(lines))
        for polygon in return_polygons:
            polygons.append(polygon)

    sets_polygon_time = timeit.time()

    print(f'reading: {loaded - start}, Line_string: {linestring - loaded},'
          f' collect polygons: {collect_polygons - linestring}, Intersectiosn: {intersections_time - collect_polygons},'
          f'Sets: {sets_time - collect_polygons}, Full_Polygons: {sets_polygon_time - sets_time},')


    final_polygons = unary_union(polygons)
    temp_polygons_list = list(final_polygons.geoms)
    polygons_list = []
    for polygon in temp_polygons_list:
        if polygon.area > 2.5e-6:
            return_polygons.append(Polygon(polygon.exterior.coords))
            polygons_list.append(Polygon(polygon.exterior.coords))
    if False:
        for polygon in polygons_list:
            gs_l = gpd.GeoSeries(polygon)
            ax = gs_l.plot()
            plt.title(f'One Final Polygon, {polygon.area = }')
            plt.show()
            plt.close('All')


    #gs_l = gpd.GeoSeries(polygons_list)
    #ax = gs_l.plot()
    #plt.title('All Polygons')
    #plt.show()
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
    final = timeit.time()

    print(f'Total: {final - start}, reading: {loaded - start}, Line_string: {linestring - loaded},'
          f' collect polygons: {collect_polygons - linestring}, Intersectiosn: {intersections_time - collect_polygons},'
          f'Sets: {sets_time - collect_polygons}, Full_Polygons: {sets_polygon_time - sets_time},'
          f' Output: {final - sets_polygon_time}')
if __name__ == '__main__':
    analyse_activities()
