import requests
import json
import requests
import geopandas as gp
from shapely.geometry import LineString


def osm_to_json(overpass_query):
    overpass_url = "http://overpass-api.de/api/interpreter"

    r = requests.get(overpass_url, params={'data': overpass_query})
    if r.status_code != 200:
        print('failed')
    r.encoding = "utf-8"
    response = json.loads(r.text)

    return response['elements']


def country_ways_query(highway_types, country_code):
    
    query_highways = "\n".join(f'way["highway"={_type}](area.searchArea);' for _type in highway_types)

    overpass_query = f"""
    [out:json][timeout:600];
    
    // gather results
    area["ISO3166-1"="""+country_code.upper()[:2]+"""]->.searchArea;
    (
      // query part for: “way”
       """ + query_highways + """
    );
    // get output
    convert way ::=::,::geom=geom(),_osm_type=type();
    out geom;
    """

    return overpass_query


def response_to_geometry(json_response):
    ways = []
    geoms = []
    for element in json_response:
        geom_ = element['geometry']['coordinates']
        geoms.append(geom_)
        type_ = element['geometry']['type']
        if type_ == 'LineString':
            way = LineString(geom_)
            ways.append(way)
        else:
            raise RuntimeError('Recieved corrupt data from the Overpass API!')

    gdf = gp.GeoDataFrame(geometry = ways, crs = 'EPSG:4326')
    gdf['geom'] = geoms

    return gdf


def get_ways(highway_types, country_code):
    overpass_query = country_ways_query(highway_types, country_code)
    response = osm_to_json(overpass_query)
    
    return response_to_geometry(response)
