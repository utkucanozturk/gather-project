import requests
import json
import requests
import geopandas as gp
from shapely.geometry import LineString


def osm_to_json(overpass_query):

    """
    Method to send html request with overpass query to overpass api interpreter
    :param: overpass_query: string of overpass query in Overpass QL
    :return: json response of query result
    """
    overpass_url = "http://overpass-api.de/api/interpreter"

    r = requests.get(overpass_url, params={'data': overpass_query})
    if r.status_code != 200:
        print('failed')
    r.encoding = "utf-8"
    response = json.loads(r.text)

    return response['elements']


def country_ways_query(highway_types, country_code):

    """
    Method to generate overpass query to get the linestrings of highways (highway can also refer in-city roads) for the country of interest
    :param: highway_types: list of tags that are associated with the highway key such as primary, secondary, residential, road, etc. See Overpass QL documentation for more.
    :param: country_code: string: Country ISO3166-1 alpha-3 code
    :return: overpass query 
    """
    
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
    
    """
    Method to transform json repsonse to geodataframe of ways
    :param: json response of query result from overpass api interpreter
    :return: geodataframe of ways
    """
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

    """
    Main method to get geodataframe of ways for a country of interest
    :param: highway_types: list of tags that are associated with the highway key such as primary, secondary, residential, road, etc. See Overpass QL documentation for more.
    :param: country_code: string: Country ISO3166-1 alpha-3 code
    :return: geodataframe of ways
    """
    overpass_query = country_ways_query(highway_types, country_code)
    response = osm_to_json(overpass_query)
    
    return response_to_geometry(response)
