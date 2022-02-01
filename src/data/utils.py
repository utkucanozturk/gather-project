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

# Define a method for displaying Earth Engine image tiles on a folium map.
def add_ee_layer(self, ee_object, vis_params, name):
    
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
    return [i[::-1] for i in geom_]


def save_map_image(map_, path, time_to_render = 5):
    img_data = map_._to_png(time_to_render)
    img = Image.open(io.BytesIO(img_data))
    img.save(path)


def get_grid(gdf, number_of_tiles):

    print('Creating grid of tiles...')

    # total area for the grid
    xmin, ymin, xmax, ymax= gdf.total_bounds
    # how many tiles across and down
    n_cells = number_of_tiles
    tile_size = (xmax-xmin)/n_cells
    # projection of the grid
    crs = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
    # create the tiles in a loop
    grid_tiles = []
    for x0 in np.arange(xmin, xmax+tile_size, tile_size ):
        for y0 in np.arange(ymin, ymax+tile_size, tile_size):
            # bounds
            x1 = x0-tile_size
            y1 = y0+tile_size
            grid_tiles.append(box(x0, y0, x1, y1)  )
    tiles = gp.GeoDataFrame(grid_tiles, columns=['geometry'], crs=crs)

    print('Visualizing the grid...')
    ax = gdf.plot(markersize=.1, figsize=(18, 12), color='red')
    plt.autoscale(False)
    tiles.plot(ax=ax, facecolor="none", edgecolor='green')
    ax.axis("off")

    print('Done!')

    return tiles


def get_polygon_area_km(geom_):

    assert geom_.type == 'Polygon', 'Input geometry should be `Polygon`!'

    geom_area = ops.transform(
        partial(
            pyproj.transform,
            pyproj.Proj(init='EPSG:4326'),
            pyproj.Proj(
                proj='aea',
                lat_1=geom_.bounds[1],
                lat_2=geom_.bounds[3]
            )
        ),
        geom_)

    # Print the area in km^2
    print('Area of the polygon is ' + str(np.sqrt(geom_area.area/1000)) + ' km^2')

    return geom_area


