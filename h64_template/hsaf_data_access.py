# -*- coding: utf-8 -*-


import os
import gzip
import shutil
from ftplib import FTP

import numpy as np
import netCDF4 as nc
import geopandas as gpd
import xarray as xr
import pandas as pd
from matplotlib.path import Path
import cartopy.feature as cfeature

import ipywidgets as widgets


class HSAFDataAccess:
    """ A class containing methods for accessing data from the H-SAF ftp server """
    

    def __init__(self):
        pass

    @staticmethod        
    def extract_gz_files(path):
        """ 
        Extracts gzipped files from a directory or a single file and deletes the original gzipped files.
    
        Parameters:
            path (str): The path of a directory or a single .gz file.
    
        Returns:
            list: List of paths to the extracted files.
        """
        extracted_files = []
    
        if os.path.isfile(path) and path.endswith('.gz'):  # Handle single .gz file
            try:
                decompressed_file = os.path.splitext(path)[0]
                with gzip.open(path, 'rb') as f_in:
                    with open(decompressed_file, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(path)  # Delete the original .gz file
                extracted_files.append(decompressed_file)
            except Exception as e:
                print(f"Error extracting data from {path}: {e}")
        elif os.path.isdir(path):  # Handle directory
            try:
                # Get list of .gz files in the directory
                gz_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.gz')]
    
                if not gz_files:
                    print("No .gz files found in the specified directory.")
                    return extracted_files
    
                for gz_file in gz_files:
                    try:
                        decompressed_file = os.path.splitext(gz_file)[0]
                        with gzip.open(gz_file, 'rb') as f_in:
                            with open(decompressed_file, "wb") as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        os.remove(gz_file)  # Delete the original .gz file
                        extracted_files.append(decompressed_file)
                    except Exception as e:
                        print(f"Error extracting data from {gz_file}: {e}")
            except Exception as e:
                raise RuntimeError(f"Error while processing directory: {e}")
        else:
            raise ValueError("Specified path must be a .gz file or directory.")
    
        return extracted_files
      

    @staticmethod
    def add_border(ax):
        """
        Add ocean and land boundaries to the given axes object.

        Parameters:
            ax (matplotlib.axes.Axes): The axes object to which boundaries are to be added.

        Returns:
            matplotlib.axes.Axes: The modified axes object.
        """

        # Define boundary types and their corresponding face colors
        boundaries = {'ocean': (173/255, 216/255, 230/255), 'land': 'none'}  # Light blue color for ocean

        try:
            for boundary_type, face_color in boundaries.items():
                feature = cfeature.NaturalEarthFeature(category='physical', name=boundary_type, scale='10m',
                                                        edgecolor='none', facecolor=face_color)
                ax.add_feature(feature)

            # Add coastlines
            ax.coastlines(resolution='10m', zorder=-1)

            return ax

        except Exception as e:
            print(f"An error occurred while adding boundaries: {e}")
            return None


    @staticmethod
    def create_netCDF_from_data(param_name, param_unit, outname, rr, lat1, lon1, datestart, dateend):
        """
        Create a NetCDF file from provided data.

        Parameters:
            param_name (str): The name for the main variable being studied.
            param_unit (str): The unit of the main variable.
            outname (str): The name for the NetCDF4 file to be created.
            rr (numpy.ndarray): The main variable data array.
            lat1 (float): The latitude coordinate.
            lon1 (float): The longitude coordinate.
            datestart (datetime.datetime): Start date of the time dimension.
            dateend (datetime.datetime): End date of the time dimension.
        """
        
        try:
            sz = rr.shape
            date_list = pd.date_range(datestart, dateend)

            ncfile = nc.Dataset(outname, mode='w', format='NETCDF4', mmap=False)
            ncfile.createDimension('time', len(date_list))  # time axis
            ncfile.createDimension('x', sz[1])  # longitude axis

            lat = ncfile.createVariable('lat', np.float32, ('x'), zlib=True)
            lat.units = 'degrees_north'
            lat.long_name = 'Latitude'
            lat[:] = lat1

            lon = ncfile.createVariable('lon', np.float32, ('x'), zlib=True)
            lon.units = 'degrees_east'
            lon.long_name = 'Longitude'
            lon[:] = lon1

            time = ncfile.createVariable('time', np.float64, ('time',), zlib=True)
            time.units = 'days since 1970-01-01'
            time.long_name = 'Time'
            time.calendar = 'standard'
            time[:] = nc.date2num(date_list.to_pydatetime(), units=time.units, calendar=time.calendar)

            main_param = ncfile.createVariable(param_name, np.float64, ('time', 'x'), zlib=True)
            main_param.units = param_unit
            main_param.standard_name = param_name 
            main_param[:] = rr

            ncfile.close()
            return "netCDF created successfully"
        except Exception as e:
            if ncfile is not None and not ncfile.closed:
                ncfile.close()  
            print(f'An error occurred: {str(e)}')
            return 'could not create netCDF from data'
        


    @staticmethod
    def points_in_polygon(query_x, query_y, vertex_x, vertex_y):
        """
        Determine whether points are inside a polygon.

        Parameters:
            query_x (numpy.ndarray): X coordinates of the query points.
            query_y (numpy.ndarray): Y coordinates of the query points.
            vertex_x (numpy.ndarray): X coordinates of the polygon vertices.
            vertex_y (numpy.ndarray): Y coordinates of the polygon vertices.

        Returns:
            numpy.ndarray: Boolean array indicating whether each query point is inside the polygon.
        """
        # Ensure inputs are numpy arrays
        query_x = np.array(query_x)
        query_y = np.array(query_y)
        vertex_x = np.array(vertex_x)
        vertex_y = np.array(vertex_y)

        shape = query_x.shape
        query_x = query_x.reshape(-1)
        query_y = query_y.reshape(-1)
        vertex_x = vertex_x.reshape(-1)
        vertex_y = vertex_y.reshape(-1)

        # Convert vertices and query points to list of tuples
        query_points = [(query_x[i], query_y[i]) for i in range(query_x.shape[0])]
        polygon_vertices = [(vertex_x[i], vertex_y[i]) for i in range(vertex_x.shape[0])]

        # Create a Path object representing the polygon
        polygon_path = Path(polygon_vertices)

        # Determine whether each query point is inside the polygon
        points_inside_polygon = polygon_path.contains_points(query_points).reshape(shape)

        return points_inside_polygon


    @staticmethod
    def points_in_shapefile(lon, lat, shapefile_path):
        """
        Checks if points are within the boundaries defined by a shapefile.

        Args:
        - lon (array-like): Array-like object containing longitudes of the points.
        - lat (array-like): Array-like object containing latitudes of the points.
        - shapefile_path (str): Path to the shapefile.

        Returns:
        - list: List containing boolean values indicating whether each point is inside the shapefile boundary.
        """


        try:
            # Read shapefile
            shapefile = gpd.read_file(shapefile_path)
    
            # Check and reproject if necessary
            if shapefile.crs != 'epsg:4326':  # Check if CRS is EPSG:4326
                shapefile = shapefile.to_crs(epsg=4326)  # Reproject to EPSG:4326
    
            # Create GeoDataFrame from points
            points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs='epsg:4326')
    
            # Check CRS and reproject if necessary
            if points.crs != shapefile.crs:
                points = points.to_crs(shapefile.crs)
    
            # Perform spatial join
            points_in_shapefile = gpd.sjoin(points, shapefile, how="inner", predicate="within")
            points_indices_inside_shapefile = points_in_shapefile.index.tolist()
    
            # Construct a boolean mask indicating whether each point is inside the shapefile
            points_in_shapefile_mask = [True if i in points_indices_inside_shapefile else False for i in range(len(lon))]
    
            return points_in_shapefile_mask
    
        except Exception as e:
            print(f'Error checking points in shapefile: {str(e)}')


    @staticmethod
    def filter_data_by_bounding_box(lon, lat, P_h60, bounding_box):
        """
        Filter data points by a bounding box.
    
        This method filters data points based on whether they fall within the specified bounding box.
        It uses the `points_in_polygon` method to determine which points are inside the bounding box.
    
        Parameters:
            lon (array-like): Array-like object containing longitudes of the data points.
            lat (array-like): Array-like object containing latitudes of the data points.
            P_h60 (array-like): Array-like object containing the data values associated with each point.
            bounding_box (tuple): A tuple containing the coordinates (x, y) defining the bounding box.
    
        Returns:
            tuple: A tuple containing three arrays: filtered_lon, filtered_lat, and filtered_P_h60.
                   These arrays contain the filtered data points that fall within the bounding box.
        """
        try:
            x, y = bounding_box
            lon_h60, lat_h60 = np.meshgrid(lat, lon, sparse=True)
            lat_h60 = np.ravel(lat_h60)
            lon_h60 = np.ravel(lon_h60)
            P_h60 = np.ravel(P_h60)
            IN = HSAFDataAccess.points_in_polygon(lon_h60, lat_h60, x, y)
            filtered_lon = lon_h60[IN]
            filtered_lat = lat_h60[IN]
            filtered_P_h60 = P_h60[IN]
            return filtered_lon, filtered_lat, filtered_P_h60
        except Exception as e:
            print(f'Error filtering data by bounding box: {str(e)}')
                                            

    @staticmethod
    def filter_data_by_shapefile(lon, lat, P_h, shapefile_path):
        """
        Filter data points by a shapefile boundary.
    
        This method filters data points based on whether they fall within the boundaries
        defined by a shapefile. It uses the GeoPandas library to read the shapefile and
        perform spatial join operations to determine which points are inside the shapefile.
    
        Parameters:
            lon (array-like): Array-like object containing longitudes of the data points.
            lat (array-like): Array-like object containing latitudes of the data points.
            P_h60 (array-like): Array-like object containing the data values associated with each point.
            shapefile_path (str): Path to the shapefile defining the boundary.
    
        Returns:
            tuple: A tuple containing three arrays: filtered_lon, filtered_lat, and filtered_P_h60.
                   These arrays contain the filtered data points that fall within the shapefile boundary.
        """
        try:
            lon, lat = np.meshgrid(lat, lon, sparse=True)
            lat_h = np.ravel(lat)
            lon_h = np.ravel(lon)
            P_h = np.ravel(P_h)
            IN = HSAFDataAccess.points_in_shapefile(lon_h, lat_h, shapefile_path)
            filtered_lon = lon_h[IN]
            filtered_lat = lat_h[IN]
            filtered_P_h = P_h[IN]
            
            # Unique values of lon and lat in the filtered set
            lon_unique = np.sort(np.unique(filtered_lon))
            lat_unique = np.sort(np.unique(filtered_lat))
            
            # Initialize P_h64_grid with NaNs
            P_h_grid = np.full((len(lat_unique), len(lon_unique)), np.nan)
            
            for i, lon in enumerate(lon_unique):
                for j, lat in enumerate(lat_unique):
                    idx = np.where((lon_h == lon) & (lat_h == lat))[0]
                    if len(idx) > 0:
                        P_h_grid[j, i] = P_h[idx[0]]
            
            
        
        
        


            return lon_unique, lat_unique, P_h_grid
        except Exception as e:
            print(f'Error filtering data by shapefile: {str(e)}')


    @staticmethod
    def cut_netcdf_by_shapefile(shapefile_path, netcdf_path):
        """
        Cut the extent of a NetCDF file to the region specified by the boundary of the given shapefile 
        and export the result as a CSV file.

        Parameters:
            shapefile_path (str): Path to the shapefile defining the region of interest.
            netcdf_path (str): Path to the NetCDF file to be cut.

        Returns:
            bool: True if the operation succeeds, False otherwise.
        """
        # Input validation
        if not shapefile_path.lower().endswith('.shp'):
            raise ValueError(shapefile_path + ' is not a shapefile.')
        if not netcdf_path.lower().endswith('.nc'):
            raise ValueError(netcdf_path + ' is not a NetCDF4 file.')

        try:
            # Open NetCDF dataset and convert to DataFrame
            ds = xr.open_dataset(netcdf_path, decode_times=True)
            df = ds.to_dataframe()

            # Read shapefile
            shp = gpd.read_file(shapefile_path)

            # Convert DataFrame to GeoDataFrame
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
            gdf.crs = shp.crs

            # Clip GeoDataFrame by shapefile boundary
            clipped_gdf = gpd.clip(gdf, shp)

            # Export clipped GeoDataFrame to CSV
            output_filename = os.path.splitext(netcdf_path)[0] + '_clipped.csv'
            clipped_gdf.to_csv(output_filename)

            return True
        except Exception as e:
            print(f'An error occurred: {str(e)}')
            return False


    @staticmethod
    def download_h64(UserName, PassWord, datestart, dateend, aoi_gdf):
        """
        Download H64 data from the FTP server, filter by AOI, and save the filtered result.

        Parameters:
            UserName (str): FTP username.
            PassWord (str): FTP password.
            datestart (str): Start date in YYYY-MM-DD format.
            dateend (str): End date in YYYY-MM-DD format.
            aoi_gdf (str): Path to shapefile defining the AOI.
        """

        # Validate date range
        try:
            pd.to_datetime(datestart)
            pd.to_datetime(dateend)
        except ValueError:
            print('Error: Invalid date format. Use YYYY-MM-DD.')
            return

        if datestart > dateend:
            print('Error: Invalid date range. Ensure datestart is less than or equal to dateend.')
            return
        
        
        # Directory to store the data
        HSAFDataAccess.create_folders()
        
        storedir = 'data'
        
        # Establish FTP connection
        try:
            with FTP('ftphsaf.meteoam.it') as ftp:
                ftp.login(user=UserName, passwd=PassWord)

                # Change directory to product category
                spt = f'h64/h64_cur_mon_data/'
                ftp.cwd(spt)

                print('Info: Preparing data download...')

                # Generate list of dates
                datelist = pd.date_range(datestart, dateend, freq='d')

                # Download files for each date
                for date in datelist:
                    filename = f'h64_{date.strftime("%Y%m%d")}_0000_24_hea.nc.gz'

                    # Check if filename is a substring of any file on the server
                    files = ftp.nlst()
                    if any(filename in file for file in files):
                        local_filename = os.path.join(storedir, filename)
                        with open(local_filename, 'wb') as loc_file:
                            ftp.retrbinary(f'RETR {filename}', loc_file.write)
                        
                        local_filename = HSAFDataAccess.extract_gz_files(local_filename)
                        
                        # Open the NetCDF file
                        with xr.open_dataset(local_filename[0], engine='netcdf4') as ds:
                           # Extract the time attribute
                           end_of_accumulation_time = ds.attrs.get('end_of_accumulation_time')
                           if end_of_accumulation_time:
                               time_coords = pd.to_datetime(end_of_accumulation_time, format='%Y%m%dT%H:%MZ')
                           else:
                               raise ValueError(f"'end_of_accumulation_time' attribute missing in file: {local_filename}")
                
                           # Assign proper coordinates to the dataset
                           ds = ds.assign_coords({
                               'latitude': ('latitude', ds.lat.values),
                               'longitude': ('longitude', ds.lon.values),
                               'time': time_coords
                           }).drop_vars(['lon', 'lat'])
                
                           # Convert the dataset into a GeoDataFrame
                           df = ds.to_dataframe().reset_index()
                           data_gdf = gpd.GeoDataFrame(
                               df, 
                               geometry=gpd.points_from_xy(df.longitude, df.latitude),
                               crs="EPSG:4326"
                           )
                
                           # Perform spatial join with AOI shapefile
                           filtered_gdf = gpd.sjoin(data_gdf, aoi_gdf, how='inner')
                           
                           filtered_df = filtered_gdf.drop(columns=['geometry','index_right'])
                           filtered_df = filtered_df.set_index(['latitude', 'longitude'])
                           filtered_ds = filtered_df.to_xarray()
                           filtered_ds = filtered_ds.assign_coords(time=filtered_df['time'].iloc[0]).assign_attrs(ds.attrs)
                           filtered_ds = filtered_ds.expand_dims(dim="time", axis=0)
                           
                           
                           
                           
                           output_file = local_filename[0].replace('.nc', '_aoi.nc')
                           filtered_ds.to_netcdf(output_file)
                          
                           # Save filtered data to a CSV file
                           csv_output_file = local_filename[0].replace('.nc', '_aoi.csv')
                           filtered_gdf.to_csv(csv_output_file, index=False)
                
                           # Clean up the original file
                        os.remove(local_filename[0])
                        print(f'Info: {os.path.basename(output_file)} downloaded')
                       
                    else:
                        print(f'Warning: No data available for {date.strftime("%d/%m/%Y")}')

                print('Info: Download completed successfully')

        except Exception as e:
            print(f'Error: An error occurred: {str(e)}')


    @staticmethod
    def download_h60(UserName, PassWord, datestart, dateend, storedir, product_category):
        """
        Download H60 data from the FTP server.

        Parameters:
            UserName (str): FTP username.
            PassWord (str): FTP password.
            datestart (str): Start date in YYYY-MM-DD format.
            dateend (str): End date in YYYY-MM-DD format.
            storedir (str): Directory to store downloaded files.
            product_category (str, optional): Product category. Defaults to 'h60'.
        """
    
        # Validate product category
        if product_category != 'h60':
            print('Error: Invalid product category. Use product category h60.')
            return

        # Validate date format
        try:
            pd.to_datetime(datestart)
            pd.to_datetime(dateend)
        except ValueError:
            print('Error: Invalid date format. Use YYYY-MM-DD.')
            return

        # Validate date range
        if datestart > dateend:
            print('Error: Invalid date range. Ensure datestart is less than or equal to dateend.')
            return

        # Validate directory path
        if not os.path.isdir(storedir):
            print('Error: Invalid directory path. Ensure storedir is a valid directory.')
            return

        # Generate list of dates
        datelist = pd.date_range(datestart, dateend, freq='d')

        # Establish FTP connection
        try:
            with FTP('ftphsaf.meteoam.it') as ftp:
                ftp.login(user=UserName, passwd=PassWord)

                # Change directory to product category
                spt = f'{product_category}/{product_category}_cur_mon_data/'
                ftp.cwd(spt)

                print('Preparing data download...')

                # Download files for each date
                files = ftp.nlst()

                for date in datelist:
                    filename = f'{product_category}_{date.strftime("%Y%m%d")}'
                    for file in files:
                        if filename in file:
                            local_filename = os.path.join(storedir, file)
                            with open(local_filename, 'wb') as loc_file:
                                ftp.retrbinary(f'RETR {file}', loc_file.write)
                            print(f'Info: {file} downloaded')
                    else:
                        print(f'Warning: No data available for {date.strftime("%d/%m/%Y")}')


                print('Info: Download completed successfully')

        except Exception as e:
            print(f'An error occurred: {str(e)}')

            
    @staticmethod
    def download_h26(UserName, PassWord, datestart, dateend, storedir, product_category):
        """
        Download H26 data from the FTP server.

        Parameters:
            UserName (str): FTP username.
            PassWord (str): FTP password.
            datestart (str): Start date in YYYY-MM-DD format.
            dateend (str): End date in YYYY-MM-DD format.
            storedir (str): Directory to store downloaded files.
            product_category (str, optional): Product category. Defaults to 'h26'.
        """
        # Validate product category
        if product_category != 'h26':
            print('Error: Invalid product category. Use product category h26.')
            return

        # Validate date format
        try:
            pd.to_datetime(datestart)
            pd.to_datetime(dateend)
        except ValueError:
            print('Error: Invalid date format. Use YYYY-MM-DD.')
            return

        # Validate date range
        if datestart > dateend:
            print('Error: Invalid date range. Ensure datestart is less than or equal to dateend.')
            return

        # Validate directory path
        if not os.path.isdir(storedir):
            print('Error: Invalid directory path. Ensure storedir is a valid directory.')
            return

        try:
            # Establish FTP connection
            with FTP('ftphsaf.meteoam.it') as ftp:
                ftp.login(user=UserName, passwd=PassWord)

                # Change directory to product category
                spt = f'{product_category}/{product_category}_cur_mon_nc/'
                ftp.cwd(spt)

                print('Info: Preparing data download...')

                # Generate list of dates
                datelist = pd.date_range(datestart, dateend, freq='d')

                # Download files for each date
                files = ftp.nlst()

                for date in datelist:
                    filename = f'{product_category}_{date.strftime("%Y%m%d")}00_R01.nc'
                    file_found = False
                    for file in files:
                        if filename in file:
                            file_found = True
                            local_filename = os.path.join(storedir, file)
                            with open(local_filename, 'wb') as loc_file:
                                ftp.retrbinary(f'RETR {file}', loc_file.write)
                            print(f'Info: {file} downloaded')
                            break  # Exit the inner loop once the file is found
                    if not file_found:
                        print(f'Warning: No data available for {date.strftime("%d/%m/%Y")}')

                print('Info: Download completed successfully')

        except Exception as e:
            print(f'Error: An error occurred: {str(e)}')

        
    @staticmethod            
    def get_lat_lon(username, psw, filename='lat_lon_0.nc.gz', local_filename='latlon.nc.gz'):
        """
        Download latitude and longitude data from the FTP server.

        Parameters:
            username (str): FTP username.
            psw (str): FTP password.
            filename (str, optional): Name of the file to download. Defaults to 'lat_lon_0.nc.gz'.
            local_filename (str, optional): Name to save the downloaded file locally. Defaults to 'latlon.nc.gz'.

        Returns:
            str: The path to the downloaded file.
        """
        try:
            # Connect to FTP server
            with FTP('ftphsaf.meteoam.it') as ftp:
                ftp.login(user=username, passwd=psw)
                ftp.cwd('utilities/matlab_code')

                # Download file
                with open(local_filename, 'wb') as loc_file:
                    if filename in ftp.nlst():
                        ftp.retrbinary(f'RETR {filename}', loc_file.write)
                        print(f'Info: {filename} downloaded')
                    else:
                        print('Warning: No positional data available')
                        return None

            # Extract gz file
            with gzip.open(local_filename, 'rb') as f_in:
                with open('lat_lon.nc', 'wb') as f_out:
                    f_out.write(f_in.read())
                
    
            print('Info: Download completed successfully')
            return os.path.join('./', 'lat_lon.nc')

        except Exception as e:
            print(f'Error: An error occurred: {str(e)}')
            return None
    

    @staticmethod
    def create_folders():
        """
        Creates necessary folders if they don't exist.
    
        The method checks if the 'output' and 'data' folders exist. If they do not exist,
        it creates them. If they exist, it deletes all files inside the 'data' folder.
    
        Parameters:
            None
    
        Returns:
            None
        """
        if not os.path.exists('output'):
            os.makedirs('output')
        if not os.path.exists('data'):
            os.makedirs('data')
        else:
            for f in os.listdir('data'):
                os.remove(os.path.join('data', f))
    

    @staticmethod
    def extract_and_clean_data(store_dir):
        """
        Extracts all gzipped files in the specified directory and removes any empty files after extraction.
    
        Parameters:
            store_dir (str): The directory containing the gzipped files to be extracted and cleaned.
    
        Returns:
            None
        """
        
        try:
            HSAFDataAccess.extract_gz_files(store_dir)
    
            # Remove empty files after extraction
            for file_name in os.listdir(store_dir):
                file_path = os.path.join(store_dir, file_name)
                if os.stat(file_path).st_size == 0:
                    os.remove(file_path)
        except Exception as e:
            print(f'Error extracting and cleaning data: {str(e)}')
    
    
    # Function to create bounding box widgets
    def create_box_widgets():
        return widgets.VBox([
            widgets.Label('Bounding Box Coordinates:'),
            widgets.FloatText(value=0.0, description='X1:'),
            widgets.FloatText(value=0.0, description='X2:'),
            widgets.FloatText(value=0.0, description='Y1:'),
            widgets.FloatText(value=0.0, description='Y2:')
        ])
    


    
    def create_shp_widgets():
        return widgets.VBox([
            widgets.Text(value='./', description= 'shp url')            
        ])

