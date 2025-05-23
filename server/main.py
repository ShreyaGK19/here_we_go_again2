from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import geopandas as gpd
import shapely.wkt
from shapely.geometry import shape, mapping
import io
import zipfile
import os
from datetime import datetime
from bson import ObjectId
import tempfile
import fiona
from fastapi.responses import PlainTextResponse

app = FastAPI(title="GIS Tool API", description="Web-based GIS tool for spatial operations")

# Add CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Configuration
uri = "mongodb+srv://aaryanwaghmare2004:Aaryan123@cluster0.6b5ubw7.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))

# Database and Collections
db = client.gis_tool
layers_collection = db.layers
operations_collection = db.operations

# Pydantic Models
class LayerMetadata(BaseModel):
    name: str
    file_type: str
    feature_count: int
    bbox: List[float]  # [minx, miny, maxx, maxy]
    crs: Optional[str] = None
    attributes: List[str]
    created_at: datetime
    file_size: int

class GISFeature(BaseModel):
    type: str = "Feature"
    geometry: Dict[str, Any]
    properties: Dict[str, Any]

class GISLayer(BaseModel):
    id: Optional[str] = None
    metadata: LayerMetadata
    features: List[GISFeature]

class OperationRequest(BaseModel):
    operation_type: str  # union, intersection, difference, buffer, spatial_join
    layer_ids: List[str]
    parameters: Optional[Dict[str, Any]] = {}

# Helper Functions
def create_geospatial_indexes():
    """Create geospatial indexes for efficient spatial queries"""
    try:
        # Create 2dsphere index on geometry for spatial operations
        layers_collection.create_index([("features.geometry", "2dsphere")])
        print("Geospatial indexes created successfully")
    except Exception as e:
        print(f"Error creating indexes: {e}")

def parse_geojson(file_content: str) -> Dict:
    """Parse GeoJSON content"""
    try:
        data = json.loads(file_content)
        if data.get('type') == 'FeatureCollection':
            return data
        elif data.get('type') == 'Feature':
            return {'type': 'FeatureCollection', 'features': [data]}
        else:
            # Single geometry
            return {
                'type': 'FeatureCollection',
                'features': [{
                    'type': 'Feature',
                    'geometry': data,
                    'properties': {}
                }]
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid GeoJSON format: {str(e)}")

def parse_shapefile_zip(file_bytes: bytes) -> Dict:
    """Parse Shapefile content from zip file"""
    try:
        # Create temporary directory for shapefile components
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract zip file
            with zipfile.ZipFile(io.BytesIO(file_bytes), 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find .shp file
            shp_files = [f for f in os.listdir(temp_dir) if f.endswith('.shp')]
            if not shp_files:
                raise HTTPException(status_code=400, detail="No .shp file found in uploaded zip")
            
            shp_path = os.path.join(temp_dir, shp_files[0])
            
            # Read with geopandas
            gdf = gpd.read_file(shp_path)
            
            # Convert to GeoJSON format
            features = []
            for _, row in gdf.iterrows():
                # Handle null geometries
                if row.geometry is None:
                    continue
                    
                # Convert properties to JSON-serializable format
                properties = {}
                for k, v in row.items():
                    if k != 'geometry':
                        # Convert numpy types to Python types
                        if hasattr(v, 'item'):
                            properties[k] = v.item()
                        elif pd.isna(v):
                            properties[k] = None
                        else:
                            properties[k] = v
                
                feature = {
                    'type': 'Feature',
                    'geometry': mapping(row.geometry),
                    'properties': properties
                }
                features.append(feature)
            
            return {
                'type': 'FeatureCollection',
                'features': features,
                'crs': str(gdf.crs) if gdf.crs else None
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing shapefile: {str(e)}")

def parse_shapefile_direct(file_bytes: bytes, filename: str) -> Dict:
    """Parse individual shapefile component"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the uploaded file
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, 'wb') as f:
                f.write(file_bytes)
            
            # Check if it's a .shp file
            if filename.lower().endswith('.shp'):
                # Try to read directly
                try:
                    gdf = gpd.read_file(file_path)
                    
                    # Convert to GeoJSON format
                    features = []
                    for _, row in gdf.iterrows():
                        # Handle null geometries
                        if row.geometry is None:
                            continue
                            
                        # Convert properties to JSON-serializable format
                        properties = {}
                        for k, v in row.items():
                            if k != 'geometry':
                                # Convert numpy types to Python types
                                if hasattr(v, 'item'):
                                    properties[k] = v.item()
                                elif pd.isna(v):
                                    properties[k] = None
                                else:
                                    properties[k] = v
                        
                        feature = {
                            'type': 'Feature',
                            'geometry': mapping(row.geometry),
                            'properties': properties
                        }
                        features.append(feature)
                    
                    return {
                        'type': 'FeatureCollection',
                        'features': features,
                        'crs': str(gdf.crs) if gdf.crs else None
                    }
                except Exception:
                    raise HTTPException(
                        status_code=400, 
                        detail="Cannot read shapefile. Please upload all required files (.shp, .shx, .dbf) as a zip archive."
                    )
            else:
                raise HTTPException(
                    status_code=400, 
                    detail="Please upload the .shp file or a zip archive containing all shapefile components."
                )
                
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing shapefile: {str(e)}")

def parse_wkt(file_content: str) -> Dict:
    """Parse WKT content"""
    try:
        lines = file_content.strip().split('\n')
        features = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Try to parse as WKT
                try:
                    geom = shapely.wkt.loads(line)
                    feature = {
                        'type': 'Feature',
                        'geometry': mapping(geom),
                        'properties': {}
                    }
                    features.append(feature)
                except:
                    # Skip invalid lines
                    continue
        
        if not features:
            raise HTTPException(status_code=400, detail="No valid WKT geometries found")
        
        return {
            'type': 'FeatureCollection',
            'features': features
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing WKT: {str(e)}")

def detect_file_type(filename: str, file_content: bytes) -> str:
    """Detect file type based on filename and content"""
    filename_lower = filename.lower()
    
    # Check by extension
    if filename_lower.endswith(('.geojson', '.json')):
        return 'geojson'
    elif filename_lower.endswith('.zip'):
        return 'shapefile_zip'
    elif filename_lower.endswith('.shp'):
        return 'shapefile'
    elif filename_lower.endswith('.wkt'):
        return 'wkt'
    
    # Try to detect by content
    try:
        # Try to parse as JSON first
        content_str = file_content.decode('utf-8')
        json.loads(content_str)
        return 'geojson'
    except:
        pass
    
    # Check if it's a zip file by magic number
    if file_content.startswith(b'PK'):
        return 'shapefile_zip'
    
    # Check if it looks like WKT
    try:
        content_str = file_content.decode('utf-8')
        if any(geom_type in content_str.upper() for geom_type in ['POINT', 'LINESTRING', 'POLYGON', 'MULTIPOINT', 'MULTILINESTRING', 'MULTIPOLYGON']):
            return 'wkt'
    except:
        pass
    
    return 'unknown'

def calculate_bbox(features: List[Dict]) -> List[float]:
    """Calculate bounding box for features"""
    if not features:
        return [0, 0, 0, 0]
    
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    
    for feature in features:
        geom = shape(feature['geometry'])
        bounds = geom.bounds
        min_x = min(min_x, bounds[0])
        min_y = min(min_y, bounds[1])
        max_x = max(max_x, bounds[2])
        max_y = max(max_y, bounds[3])
    
    return [min_x, min_y, max_x, max_y]

def get_attributes(features: List[Dict]) -> List[str]:
    """Extract attribute names from features"""
    attributes = set()
    for feature in features:
        attributes.update(feature.get('properties', {}).keys())
    return list(attributes)

# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize database indexes on startup"""
    try:
        client.admin.command('ping')
        print("Connected to MongoDB successfully!")
        create_geospatial_indexes()
    except Exception as e:
        print(f"MongoDB connection error: {e}")

@app.get("/")
async def root():
    return {"message": "GIS Tool API is running"}

@app.post("/upload-layer")
async def upload_layer(
    file: UploadFile = File(...),
    layer_name: str = Form(...)
):
    """Upload and store a GIS layer (GeoJSON, Shapefile, or WKT)"""
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_content = await file.read()
    file_size = len(file_content)
    
    # Detect file type
    file_type = detect_file_type(file.filename, file_content)
    
    # Parse based on detected file type
    try:
        if file_type == 'geojson':
            geojson_data = parse_geojson(file_content.decode('utf-8'))
            crs = geojson_data.get('crs')
            original_format = 'geojson'
        elif file_type == 'shapefile_zip':
            geojson_data = parse_shapefile_zip(file_content)
            crs = geojson_data.get('crs')
            original_format = 'shapefile'
        elif file_type == 'shapefile':
            geojson_data = parse_shapefile_direct(file_content, file.filename)
            crs = geojson_data.get('crs')
            original_format = 'shapefile'
        elif file_type == 'wkt':
            geojson_data = parse_wkt(file_content.decode('utf-8'))
            crs = None
            original_format = 'wkt'
        else:
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file format. Supported formats: GeoJSON (.geojson, .json), Shapefile (.shp or .zip), WKT (.wkt)"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
    
    features = geojson_data['features']
    
    if not features:
        raise HTTPException(status_code=400, detail="No valid features found in the uploaded file")
    
    # Create metadata
    metadata = LayerMetadata(
        name=layer_name,
        file_type=original_format,
        feature_count=len(features),
        bbox=calculate_bbox(features),
        crs=crs,
        attributes=get_attributes(features),
        created_at=datetime.now(),
        file_size=file_size
    )
    
    # Store in MongoDB
    layer_doc = {
        'metadata': metadata.dict(),
        'features': features
    }
    
    result = layers_collection.insert_one(layer_doc)
    
    return {
        "message": "Layer uploaded successfully",
        "layer_id": str(result.inserted_id),
        "metadata": metadata.dict(),
        "converted_format": "GeoJSON" if original_format != 'geojson' else None
    }

@app.get("/layers")
async def get_layers():
    """Get all layers with metadata"""
    layers = []
    for doc in layers_collection.find({}, {'features': 0}):  # Exclude features for performance
        doc['id'] = str(doc['_id'])
        del doc['_id']
        layers.append(doc)
    return {"layers": layers}

@app.get("/layers/{layer_id}")
async def get_layer(layer_id: str):
    """Get specific layer with all features"""
    try:
        doc = layers_collection.find_one({'_id': ObjectId(layer_id)})
        if not doc:
            raise HTTPException(status_code=404, detail="Layer not found")
        
        doc['id'] = str(doc['_id'])
        del doc['_id']
        return doc
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid layer ID: {str(e)}")

@app.get("/layers/{layer_id}/geojson")
async def get_layer_as_geojson(layer_id: str):
    """Get specific layer as GeoJSON format"""
    try:
        doc = layers_collection.find_one({'_id': ObjectId(layer_id)})
        if not doc:
            raise HTTPException(status_code=404, detail="Layer not found")
        
        return {
            "type": "FeatureCollection",
            "features": doc['features'],
            "metadata": doc['metadata']
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid layer ID: {str(e)}")

@app.delete("/layers/{layer_id}")
async def delete_layer(layer_id: str):
    """Delete a layer"""
    try:
        result = layers_collection.delete_one({'_id': ObjectId(layer_id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Layer not found")
        return {"message": "Layer deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid layer ID: {str(e)}")

@app.post("/operations/spatial-query")
async def spatial_query(
    layer_id: str,
    geometry: Dict[str, Any],
    operation: str = "intersects"  # intersects, within, contains
):
    """Perform spatial query on a layer"""
    try:
        # MongoDB geospatial query
        query = {}
        if operation == "intersects":
            query = {"features.geometry": {"$geoIntersects": {"$geometry": geometry}}}
        elif operation == "within":
            query = {"features.geometry": {"$geoWithin": {"$geometry": geometry}}}
        
        # Find matching features
        layer_doc = layers_collection.find_one(
            {'_id': ObjectId(layer_id)},
            query
        )
        
        if not layer_doc:
            return {"features": []}
        
        return {"features": layer_doc.get('features', [])}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Spatial query error: {str(e)}")

@app.get("/layers/{layer_id}/bbox")
async def get_layer_bbox(layer_id: str):
    """Get bounding box for a layer"""
    try:
        doc = layers_collection.find_one(
            {'_id': ObjectId(layer_id)}, 
            {'metadata.bbox': 1}
        )
        if not doc:
            raise HTTPException(status_code=404, detail="Layer not found")
        
        return {"bbox": doc['metadata']['bbox']}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid layer ID: {str(e)}")

@app.post("/operations/buffer")
async def buffer_operation(
    layer_id: str,
    distance: float,
    result_layer_name: str
):
    """Create buffer around features in a layer"""
    try:
        # Get original layer
        layer_doc = layers_collection.find_one({'_id': ObjectId(layer_id)})
        if not layer_doc:
            raise HTTPException(status_code=404, detail="Layer not found")
        
        # Process features with buffer
        buffered_features = []
        for feature in layer_doc['features']:
            geom = shape(feature['geometry'])
            buffered_geom = geom.buffer(distance)
            
            buffered_feature = {
                'type': 'Feature',
                'geometry': mapping(buffered_geom),
                'properties': feature['properties'].copy()
            }
            buffered_features.append(buffered_feature)
        
        # Create new layer with buffered features
        metadata = LayerMetadata(
            name=result_layer_name,
            file_type="buffer_result",
            feature_count=len(buffered_features),
            bbox=calculate_bbox(buffered_features),
            crs=layer_doc['metadata'].get('crs'),
            attributes=get_attributes(buffered_features),
            created_at=datetime.now(),
            file_size=0
        )
        
        new_layer_doc = {
            'metadata': metadata.dict(),
            'features': buffered_features
        }
        
        result = layers_collection.insert_one(new_layer_doc)
        
        return {
            "message": "Buffer operation completed",
            "result_layer_id": str(result.inserted_id),
            "feature_count": len(buffered_features)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Buffer operation error: {str(e)}")

@app.post("/upload-shp-xml")
async def upload_shp_xml(
    file: UploadFile = File(...),
    layer_name: str = Form(...)
):
    """
    Upload and store a .shp.xml metadata file in MongoDB.
    """
    if not file.filename or not file.filename.lower().endswith('.xml'):
        raise HTTPException(status_code=400, detail="Please upload a .shp.xml file.")
    file_content = await file.read()
    try:
        xml_text = file_content.decode('utf-8')
        doc = {
            "layer_name": layer_name,
            "filename": file.filename,
            "xml_content": xml_text,
            "uploaded_at": datetime.now()
        }
        result = db.shp_xml_metadata.insert_one(doc)
        return {
            "message": ".shp.xml metadata uploaded successfully",
            "metadata_id": str(result.inserted_id)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing XML: {str(e)}")

@app.get("/shp-xml/{metadata_id}", response_class=PlainTextResponse)
async def get_shp_xml(metadata_id: str):
    """
    Retrieve a .shp.xml metadata file from MongoDB by its ID.
    """
    from bson.errors import InvalidId
    try:
        doc = db.shp_xml_metadata.find_one({"_id": ObjectId(metadata_id)})
        if not doc:
            raise HTTPException(status_code=404, detail="Metadata not found")
        return doc["xml_content"]
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid metadata ID")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error retrieving XML: {str(e)}")

@app.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats"""
    return {
        "supported_formats": [
            {
                "format": "GeoJSON",
                "extensions": [".geojson", ".json"],
                "description": "Geographic JavaScript Object Notation"
            },
            {
                "format": "Shapefile",
                "extensions": [".shp", ".zip"],
                "description": "ESRI Shapefile (upload .shp directly or all components as .zip)"
            },
            {
                "format": "WKT",
                "extensions": [".wkt"],
                "description": "Well-Known Text"
            }
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        client.admin.command('ping')
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": "disconnected", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)