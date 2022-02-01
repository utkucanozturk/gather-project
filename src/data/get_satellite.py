import json
import geopandas as gp
import ee
import geemap


def gee_init(json_key_path = None, service_account = None):
    """
    Authenticate & Initialize the GEE API
    :param json_key_path :string: path to GEE API key json file
    :param service_account :string: service account
    """

    print('Initializing the Google Earth Engine API...')
    try:
        ee.Initialize()
        print('Initialized!')
    except:
        if json_key_path is None or service_account is None:
                print('Please authenticate manually...')
                ee.Authenticate()
                ee.Initialize()
        else:
            try:
                credentials = ee.ServiceAccountCredentials(service_account, json_key_path)
                ee.Initialize(credentials)
                print('Initialized!')
            except:
                raise RuntimeError('Unable to initialize Googe Earth Engine API!')


def create_feature_collection(geometries):
    """
    Create feature collection from multiple geometries
    :param geometries: N size geometry object
    :return : GEE feature collection of size N
    """

    roi = gp.GeoSeries(geometries).to_json()
    res = json.loads(roi)
    feature_collection = geemap.geojson_to_ee(res, geodesic=True)
    return feature_collection


def maskS2clouds(image):
    """
    The cloud masking function provided by GEE but adapted for use in Python.
    :param image: gee image object
    :return : masked image that matches the cloud-free conditions
    """

    qa = image.select('QA60') #QA60 refers to band of the cloud coverage

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    mask = mask.bitwiseAnd(cirrusBitMask).eq(0)

    return image.updateMask(mask).divide(10000)


def roi_by_country(country_name):
    """
    Get the geometry of the specified country
    :param country_name: US-recognized country name, see https://developers.google.com/earth-engine/datasets/catalog/USDOS_LSIB_SIMPLE_2017
    :return : region of interest as gee geometry object
    """

    # Get a feature collection of administrative boundaries.
    countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    
    # Filter the feature collection to country of interest.
    roi = countries.filter(ee.Filter.eq('county_na', country_name))

    return roi
    

def Image_Processing(image_collection, start_date, end_date, feature_collection = None, country_name = None):
    """
    Image processing function for the GEE satellite imagery
    :param image_collection: one of GEE image collection dataset names e.g. 'COPERNICUS/S2_SR'
    :param start_date: start date of the period for which image is wanted to be used in format 'YYYY-MM-DD'
    :param end_date: end date of the period for which image is wanted to be used in format 'YYYY-MM-DD'
    :param band: band(layer) name(s) that will be selected from the image (None if )
    :param feature_collection: 
    :param country_name: US-recognized country name, see https://developers.google.com/earth-engine/datasets/catalog/USDOS_LSIB_SIMPLE_2017
    :return : gee image object
    """

    print('Processing image...')

    image_col = ee.ImageCollection(image_collection)

    if country_name is not None:
        roi = roi_by_country(country_name)
        image = image_col.filterBounds(roi)
    elif feature_collection is not None:
        image = image_col.filterBounds(feature_collection)
    else:
        raise RuntimeError('At least one of the feature_collection or country_name must be specified!')

    image = image.filterDate(start_date, end_date)
    
    least_cloudy = ee.Image(image.sort('CLOUD_COVER').first())

    print('Cloud Cover (%):', least_cloudy.get('CLOUD_COVER').getInfo())

    print('Image processing complete!')

    return least_cloudy