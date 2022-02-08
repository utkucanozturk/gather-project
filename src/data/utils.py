import folium
import ee
import io
from PIL import Image
import pyproj    
import shapely.ops as ops
from shapely.geometry import box
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gp


def add_ee_layer(self, ee_object, vis_params, name):

    """
    Method for displaying Earth Engine image tiles on a folium map
    :param: ee_object: image collected from google earth engine api
    :param: vis_params: dictionary of visual parameters such as image bands, gamma
    :param: name:
    """

    try:    
        # display ee.Image()
        if isinstance(ee_object, ee.image.Image):    
            map_id_dict = ee.Image(ee_object).getMapId(vis_params)
            folium.raster_layers.TileLayer(
            tiles = map_id_dict['tile_fetcher'].url_format,
            attr = 'Google Earth Engine',
            name = name,
            overlay = True,
            control = True
            ).add_to(self)
        # display ee.ImageCollection()
        elif isinstance(ee_object, ee.imagecollection.ImageCollection):    
            ee_object_new = ee_object.mosaic()
            map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
            folium.raster_layers.TileLayer(
            tiles = map_id_dict['tile_fetcher'].url_format,
            attr = 'Google Earth Engine',
            name = name,
            overlay = True,
            control = True
            ).add_to(self)
        # display ee.Geometry()
        elif isinstance(ee_object, ee.geometry.Geometry):    
            folium.GeoJson(
            data = ee_object.getInfo(),
            name = name,
            overlay = True,
            control = True
        ).add_to(self)
        # display ee.FeatureCollection()
        elif isinstance(ee_object, ee.featurecollection.FeatureCollection):  
            ee_object_new = ee.Image().paint(ee_object, 0, 2)
            map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
            folium.raster_layers.TileLayer(
            tiles = map_id_dict['tile_fetcher'].url_format,
            attr = 'Google Earth Engine',
            name = name,
            overlay = True,
            control = True
        ).add_to(self)
    
    except:
        print("Could not display {}".format(name))


def reverse_coord(geom_):

    """
    Method to reverse list of point lat-lon pairs. E.g. [lon, lat] to [lat, lon] or [lat, lon] to [lon, lat]
    :param: geom_: list of point lat-lon pairs [lon, lat] or [lat, lon]
    :return: list of point lat-lon pairs
    """

    return [i[::-1] for i in geom_]


def save_map_image(map_, path, time_to_render = 5):

    """
    Method to save folium map as image.
    :param: map_: Folium map object
    :param: path: path (must include file name with extension e.g. ./image.png, ./image.jpg) where the image file to be created. 
    :param: time_to_render: seconds to render the folium map image. Note that low numbers might result in empty images!
    """

    img_data = map_._to_png(time_to_render)
    img = Image.open(io.BytesIO(img_data))
    img.save(path)


def get_grid(gdf, tile_size, visualize = False):

    """
    Method to get the grid of tiles for the area covered by instances of a geodataframe
    :param: gdf: geopandas geodataframe object
    :param: tile_size: size of the edge of each tile
    :return: geodataframe of tiles
    """

    print('Creating grid of tiles...')

    # projection of the grid
    crs = 'EPSG:3395'

    gdf = gdf.to_crs(crs = crs)

    # total area for the grid
    xmin, ymin, xmax, ymax= gdf.total_bounds

    tile_size = tile_size * 1000
    
    # create the tiles in a loop
    grid_tiles = []
    for x0 in np.arange(xmin, xmax + tile_size, tile_size ):
        for y0 in np.arange(ymin, ymax + tile_size, tile_size):
            # bounds
            x1 = x0 - tile_size
            y1 = y0 + tile_size
            grid_tiles.append(box(x0, y0, x1, y1)  )
    tiles = gp.GeoDataFrame(grid_tiles, columns=['geometry'], crs = crs)

    tiles = gp.sjoin(tiles, gdf, how = 'inner', predicate='intersects')

    if visualize:
        print('Visualizing the grid...')
        ax = gdf.plot(markersize=.1, figsize=(18, 12), color='indigo')
        plt.autoscale(False)
        tiles.plot(ax=ax, facecolor="none", edgecolor='lime')
        ax.axis("off")

    print('Done!')

    return tiles.to_crs(crs = 'EPSG:4326')

