def extract_object_properties(segmented_image_path, intensity_image_path, image_name, xy_scale, z_scale):
    """
    Takes a segmented image and the corresponding intensity image for that segmentation

    Measures minimum, mean, max, and total intensity in addition to total # of pixels per object

    Returns a list of dictionaries, where each dictionary has a key-value pair mapping
    object property terms to their values
    e.g. 'area': 163
    """

    print('Extracting object properties for {image_name}'.format(image_name=image_name))

    # import packages needed for object extraction
    from skimage.io import imread
    from scipy.ndimage import label as ndi_label
    from skimage import measure

    # read in images
    segmented_image = imread(segmented_image_path)
    intensity_image = imread(intensity_image_path)

    # label connected components
    labeled, num_features = ndi_label(segmented_image)

    # measure properties
    region_properties = measure.regionprops(labeled, intensity_image = intensity_image)

    object_data_list = []

    for prop in region_properties:

        # apply the z scale and xy scales to the centroid and coordinates lists
        centroid = list(prop.centroid)
        centroid_scaled = [centroid[0] * z_scale, centroid[1]*xy_scale, centroid[2] * xy_scale]

        coords = prop.coords.tolist()
        coords_scaled = [[coord[0]*z_scale, coord[1]* xy_scale, coord[2]*xy_scale] for coord in coords ]

        # create a dict containing object properties
        object_properties_dict = {
        'area': int(prop.area),
        'min_intensity' : int(prop.min_intensity),
        'max_intensity' : int(prop.max_intensity),
        'mean_intensity' : int(prop.mean_intensity),
        'total_intensity': int(prop.intensity_image.sum()),
        'object_id' : int(prop.label),
        'name': image_name,
        'centroid': centroid_scaled,
        'coordinates': coords_scaled,
        'intensity_image': prop.intensity_image.tolist()}

        object_data_list.append(object_properties_dict)

    return object_data_list


def insert_object_data(structure, object_data_list, database_name, db_user, db_password, db_host):
    """ Inputs: structure (string that describes the subcellular structure)
                object_data_list - a list that is the output of the extract_object_properties function,
                a psycopg2 connection object for connecting to the experiment's database

        Inserts data from the object_data_list into the structure table.
        Use caution: this funtion appends data if the structure table already contains object data for that image

        Returns nothing
    """

    import psycopg2
    from psycopg2 import sql

    conn = psycopg2.connect(database=database_name, user=db_user, password=db_password, host=db_host)
    cur = conn.cursor()

    query = sql.SQL("""INSERT into {table}
                    (area, min_intensity, max_intensity, mean_intensity, total_intensity, object_id, name, centroid, coordinates, raw_img)
                    VALUES
                    (%(area)s, %(min_intensity)s, %(max_intensity)s, %(mean_intensity)s, %(total_intensity)s, %(object_id)s, %(name)s, %(centroid)s, %(coordinates)s, %(intensity_image)s);""").format(
                        table=sql.Identifier(structure))


    cur.executemany(query, object_data_list)
    conn.commit()

    cur.close()
    conn.close()

    return

def extract_surface_coordinates(coordinates_list):
    """ Input: a list of coordinates that define one object

    Logic is to include all coordinates in the minimum and maximum z slices
    Then hold z, x constant and include the coordinates w/ the min and max y values for each z,x pair
    Then hold z, y constant and include the coordinates w/ the min and max x values for each z,y pair

    Return list of exterior coordinates, excluding any invaginations
    """
    z_coords = set([coord[0] for coord in coordinates_list])

    surface_coords = []

    # add all of the coordinates that are in the minimum and maximum z slices
    z_min = min(z_coords)
    z_max = max(z_coords)

    minimum_z_coords = [coord for coord in coordinates_list if coord[0] == z_min]
    maximum_z_coords = [coord for coord in coordinates_list if coord[0] == z_max]

    surface_coords.extend(minimum_z_coords)
    surface_coords.extend(maximum_z_coords)

    # now iterate over z coordinates
    # hold x constant and add [z, x, y_min] and [z, x, y_max] if not already in surface list
    # then hold y constant and add [z, x_min, y] and [z, x_max, y] if not already in surface list

    for z in z_coords:

        select_xy_coords = [(coord[1:]) for coord in coordinates_list if coord[0] == z]

        x_coords = set([coord[0] for coord in select_xy_coords])
        y_coords = set([coord[1] for coord in select_xy_coords])

        for x in x_coords:
            select_y_values = [xy_coord[1] for xy_coord in select_xy_coords if xy_coord[0] == x]

            min_y = min(select_y_values)

            surface_coords.append([z, x, min_y])

            max_y = max(select_y_values)
            surface_coords.append([z, x, max_y])

        for y in y_coords:
            select_x_values = [xy_coord[0] for xy_coord in select_xy_coords if xy_coord[1] == y]

            min_x = min(select_x_values)
            surface_coords.append([z, min_x, y])

            max_x = max(select_x_values)
            surface_coords.append([z, max_x, y])

    # remove duplicates
    duplicates_removed = list((set(map(tuple, surface_coords))))

    return duplicates_removed

def centroid_measurements_closest_structure_2(centroid_1, structure_2_centroid_data):
    """ structure_2 centroid data is a list of tuples in format [(id, centroid)] w/ all the centroids for the subcellular target of interest for that image
    """

    # package import
    import numpy as np

    # convert centroid to numpy array
    centroid_1 = np.array(centroid_1)

    # measure distances between all centroids
    distance_data = []
    for id_centroid_row in structure_2_centroid_data:
        table_id = id_centroid_row[0]
        centroid_2 = np.array(id_centroid_row[1])

        distance = np.linalg.norm(centroid_1 - centroid_2)

        distance_data.append({'structure_2_id': table_id, 'distance': distance})

    sorted_distances = sorted(distance_data, key=lambda k: k['distance'])[0:3]

    closest_structure_2_ids = tuple([structure_dict['structure_2_id'] for structure_dict in sorted_distances])

    return closest_structure_2_ids

def minimum_distance(object_1, object_2):
    """ Takes two lists as input
    A list of numpy arrays of coordinates that make up object 1 and object 2
    Measures the distances between each of the coordinates
    Returns the minimum distance between the two objects, as calculated using a vector norm
    Stops the calculation and returns 0 if two coordinates overlap
    """

    # package import
    import numpy as np

    # main algorithm
    minimum_distance = 100000

    for coord_1 in object_1:
        for coord_2 in object_2:
            distance_btwn_coords = np.linalg.norm(coord_1 - coord_2)
            if distance_btwn_coords == 0:
                minimum_distance = distance_btwn_coords
                return float(minimum_distance)
            elif distance_btwn_coords < minimum_distance:
                minimum_distance = distance_btwn_coords

    return float(minimum_distance)


def create_postgres_table(conn, structure):
    """ Function to create foundational table for holding data related to images
    All images of subcellular structures have this foundational data collected and stored
    The rna table has specialized additional columns for distance measurements that are added separately

    Input: connection object for psycopg2.connect, string describing the subcellular structure this table is for
    (e.g. 'rna' or 'centrosomes')

    Output: creates a table in the database that the connection object connects to.
    Closes the cursor and returns nothing

    """

    import psycopg2
    from psycopg2 import sql

    #create a connection object using the connect_postgres() function and initialize a cursor

    cursor = conn.cursor()

    # Create a tuple containing the SQL commands for creating the tables
    SQL_smFISH_command = """)"""

    cursor.execute(sql.SQL("""

        CREATE TABLE IF NOT EXISTS {table}
            (id SERIAL NOT NULL,
            object_id INT NOT NULL,
            name TEXT NOT NULL,
            area REAL,
            min_intensity REAL,
            max_intensity REAL,
            mean_intensity REAL,
            total_intensity REAL,
            centroid REAL [][][],
            coordinates REAL [][][],
            raw_img INT [][][]);""").format(
                                table=sql.Identifier(structure)))


    conn.commit()
    cursor.close()

    return

def add_distance_columns(structure_1, structure_2, conn):
    # package import
    from psycopg2 import sql

    # create variables that contain the names for the new columns
    distance_to_structure_2 = 'distance_to_' + structure_2
    structure_2_id = structure_2 + '_id'

    # update the database
    cur = conn.cursor()

    query = sql.SQL("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col_1} REAL, ADD COLUMN IF NOT EXISTS {col_2} INT;").format(
                            table=sql.Identifier(structure_1),
                            col_1=sql.Identifier(distance_to_structure_2),
                            col_2=sql.Identifier(structure_2_id))

    cur.execute(query)
    conn.commit()
    cur.close()

    return


def test_data_db(image_name, structure, database_name, db_user, db_password, db_host):
    """Inputs: string describing the name of an image and a string describing a subcellular structure in that image
    and a connection object

    Tests if any object data for a given structure and image has been inserted into the database

    Returns 'True' if there is 1 or more objects for that structure are in the db
    Returns 'False' if there are no objects for that structure are in the db

    """

    import psycopg2
    from psycopg2 import sql

    conn = psycopg2.connect(database=database_name, user=db_user, password=db_password, host=db_host)
    cur = conn.cursor()
    query = sql.SQL("SELECT * FROM {table} where name = %s").format(
                            table=sql.Identifier(structure))

    cur.execute(query, (image_name,))

    processed_bool = bool(cur.fetchone())

    cur.close()

    return processed_bool

def measure_distance_update_db(structure_1_data_list, structure_1, structure_2, database_name, db_user, db_password, db_host):

    import numpy as np
    import psycopg2
    from psycopg2 import sql

    distance_col = 'distance_to_' + structure_2

    for structure_1_data in structure_1_data_list:
        # unpack structure 1 data
        image_name = structure_1_data[0]
        structure_1_id = structure_1_data[1]
        centroid_1 = np.array(structure_1_data[2])
        coords_1 = structure_1_data[3]

        # get all structure 2 centroids for that image
        conn = psycopg2.connect(database=database_name, user=db_user, password=db_password, host=db_host)
        cur = conn.cursor()

        structure_2_centroid_query = sql.SQL("""SELECT id, centroid
                                    FROM {structure_2}
                                    WHERE name = %(image_name)s""").format(
                            structure_2=sql.Identifier(structure_2))

        cur.execute(structure_2_centroid_query, {'image_name': image_name})

        structure_2_centroid_data = cur.fetchall()

        cur.close()
        conn.close()

        # measure centroid to centroid distances
        closest_structure_2 = centroid_measurements_closest_structure_2(centroid_1, structure_2_centroid_data)

        # now get the coordinates for those structure 2 ids
        structure_2_coords_query = sql.SQL("SELECT id, coordinates FROM {structure_2} WHERE id IN %(id)s;").format(
                                            structure_2=sql.Identifier(structure_2))

        conn = psycopg2.connect(database=database_name, user=db_user, password=db_password, host=db_host)
        cur = conn.cursor()

        cur.execute(structure_2_coords_query, {'id': closest_structure_2})

        structure_2_coord_data = cur.fetchall()
        cur.close()
        conn.close()

        # prepare the coordinates for object 1 for distance measurements

        surface_coords_1 = extract_surface_coordinates(coords_1)
        surface_coords_1 = [np.array(coord) for coord in surface_coords_1]

        closest_structure_2_distance = 100000
        closest_structure_2_id = None

        # now iterate over structure 2 coords
        for id_coord_row in structure_2_coord_data:
            structure_2_id = id_coord_row[0]
            coords_2 = id_coord_row[1]

            surface_coords_2 = extract_surface_coordinates(coords_2)
            surface_coords_2 = [np.array(coord) for coord in surface_coords_2]

            distance_to_structure_1 = minimum_distance(surface_coords_1, surface_coords_2)

            if distance_to_structure_1 < closest_structure_2_distance:
                closest_structure_2_distance = distance_to_structure_1
                closest_structure_2_id = structure_2_id

        # now update the database

        update_distance_query = sql.SQL("""UPDATE {structure_1}
                                            SET {distance_col} = %(closest_structure_2_distance)s,
                                            {structure_2_id} = %(closest_structure_2_id)s
                                            WHERE id = %(structure_1_id)s;""").format(
                                    structure_1=sql.Identifier(structure_1),
                                    distance_col=sql.Identifier(distance_col),
                                    structure_2_id=sql.Identifier(structure_2 + '_id'))

        conn = psycopg2.connect(database=database_name, user=db_user, password=db_password, host=db_host)
        cur = conn.cursor()

        cur.execute(update_distance_query, {'closest_structure_2_distance': closest_structure_2_distance,
                                           'closest_structure_2_id': closest_structure_2_id,
                                            'structure_1_id': structure_1_id})
        conn.commit()

        cur.close()
        conn.close()

    return None


def measure_distance_by_obj(structure_1, structure_2, parallel_processing_bool, database_name, db_user, db_password, db_host):
    import psycopg2
    from psycopg2 import sql
    import multiprocessing as mp

    # first get all structure 1 objects that have not been measured from the database

    distance_col = 'distance_to_' + structure_2

    # first get all null values
    structure_1_data_query = sql.SQL("""SELECT name, id, centroid, coordinates
                            FROM {structure_1}
                            WHERE {distance_col} IS NULL;""").format(
                    structure_1=sql.Identifier(structure_1),
                    distance_col=sql.Identifier(distance_col))

    conn = psycopg2.connect(database=database_name, user=db_user, password=db_password, host=db_host)
    cur = conn.cursor()

    cur.execute(structure_1_data_query)
    structure_1_data_all = cur.fetchall()

    cur.close()
    conn.close()

    null_count = len(structure_1_data_all)

    if null_count == 0:
        print('All object distances between {structure_1} and {structure_2} have been measured'.format(
                                        structure_1=structure_1,
                                        structure_2=structure_2))

    else:
        if not parallel_processing_bool:
            print('Measuring distances between {structure_1} and {structure_2} without parallel processing'.format(
                                        structure_1=structure_1,
                                        structure_2=structure_2))
            measure_distance_update_db(structure_1_data_all, structure_1, structure_2, database_name, db_user, db_password, db_host)

        else:
            print('Measuring distances between {structure_1} and {structure_2} with parallel processing'.format(
                                        structure_1=structure_1,
                                        structure_2=structure_2))
            # get cpu count and estimate the number of objects each processor should process
            cpu_count = mp.cpu_count() - 1
            number_per_processor = round(null_count/cpu_count) + 1

            # divide the structure_1_data list into smaller pieces
            structure_1_chunks = chunk_list(structure_1_data_all, number_per_processor)

            # make a list of tuples containing these different subset lists
            argument_tuples = []
            for subset_list in structure_1_chunks:
                argument_tuple = (subset_list, structure_1, structure_2, database_name, db_user, db_password, db_host)
                argument_tuples.append(argument_tuple)

            pool = mp.Pool(cpu_count)
            result = pool.starmap(measure_distance_update_db, argument_tuples)
    return None

def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""

    chunks = []

    for i in range(0, len(lst), n):
        chunks.append(lst[i:i + n])

    return chunks

def delete_data_db(image_name, structure, conn):
    """Inputs: string describing the name of an image and a string describing a subcellular structure in that image
    and a connection object

    Deletes all the data in the structure table for that name; use this to re-process data / avoid
    appending duplicate objects

    Returns None
    """

    from psycopg2 import sql

    cur = conn.cursor()
    query = sql.SQL("DELETE FROM {table} where name = %s").format(
                            table=sql.Identifier(structure))

    cur.execute(query, (image_name,))
    conn.commit()

    cur.close()

    return

def calculate_percent_distributions(image_name, image_name_column, structure_1, structure_2, distances, conn):
    """ Takes an image name (string), a list of distances, and a connection object

    Measures the percent of total RNA (percent_total_rna) and the percent of total RNA in granules (percent_granule_rna) at each distance in the distances list

    Returns a dict mapping the image name, the distances, the percent_total_rna, and the percent RNA in granules
    """
    # package import
    from psycopg2 import sql
    import psycopg2

    cur = conn.cursor()

    distance_col = 'distance_to_' + structure_2

    rna_total_query = sql.SQL("""SELECT SUM(total_intensity) FROM {rna_table}
                WHERE {name} = %(image_name)s AND {distance_col} <= %(max_distance)s""").format(
                rna_table = sql.Identifier(structure_1),
                distance_col =sql.Identifier(distance_col),
                name = sql.Identifier(image_name_column))

    cur.execute(rna_total_query, {'image_name':image_name, 'max_distance' : distances[-1]})
    total_rna = cur.fetchall()[0][0]

    percent_rna_list = []
    percent_granule_list = []

    for distance in distances:
        rna_sum_query = sql.SQL("""SELECT SUM(total_intensity) from {rna_table}
                WHERE {name} = %(image_name)s AND {distance_col} <= %(distance)s""").format(
                rna_table = sql.Identifier(structure_1),
                distance_col =sql.Identifier(distance_col),
                name = sql.Identifier(image_name_column))

        cur.execute(rna_sum_query, {'image_name':image_name, 'distance' : distance})
        sum_rna = cur.fetchall()[0][0]

        rna_granule_query = sql.SQL("""SELECT SUM(total_intensity) from {rna_table}
                WHERE {name} = %(image_name)s AND {distance_col} <= %(distance)s
                AND normalized_intensity >= 4""").format(
                rna_table = sql.Identifier(structure_1),
                distance_col =sql.Identifier(distance_col),
                name = sql.Identifier(image_name_column))

        cur.execute(rna_granule_query, {'image_name':image_name, 'distance' : distance})
        granule_rna = cur.fetchall()[0][0]

        # avoid errors due to being unable to divide "None" by a number
        if sum_rna == None:
            sum_rna = 0

        if granule_rna == None:
            granule_rna = 0

        percent_total_rna = sum_rna / total_rna * 100
        percent_granule_rna = granule_rna / total_rna * 100

        percent_rna_list.append(percent_total_rna)
        percent_granule_list.append(percent_granule_rna)

    cur.close()

    rna_distribution_dict = {'distance': distances, 'percent_total_rna': percent_rna_list, 'percent_granule_rna' : percent_granule_list}

    return rna_distribution_dict

def calculate_distributions_by_image(image_data_list, distance_threshold, step_size, image_name_column, structure_1, structure_2, conn):
    """ Takes a list of dictionaries containing image metadata including the image name under the key 'name', a distance threshold, a step_size, the structure_2 distance target, and a postgres db connection object

    Calculates the percent of total RNA and the percent of RNA in granules from 0 microns to the distance threshold away from a structure_2 object (or max image distance if the distance_threshold is set to None) at increments dictated by the step size

    Returns a pandas dataframe object containing the distribution data
    """

    # package import
    import psycopg2
    from psycopg2 import sql
    import numpy as np
    import pandas as pd

    cur = conn.cursor()

    distribution_dicts = []

    for image_data_dict in image_data_list:

        image_name = image_data_dict[image_name_column]
        print('Calculating cumulative distributions for ' + image_name)

        # if user specifies distance_threshold = None, then get the max distance to structure_2 from the rna database
        if distance_threshold == None:
            distance_col = 'distance_to_' + structure_2
            max_distance_query = sql.SQL("SELECT MAX({distance_col}) from {rna_table} WHERE {name} = %(image_name)s").format(rna_table=sql.Identifier(structure_1), distance_col = sql.Identifier(distance_col), name = sql.Identifier(image_name_column))
            cur.execute(max_distance_query, {'image_name':image_name})
            distance_threshold = cur.fetchall()[0][0]

        distances = np.arange(0, distance_threshold, step_size)
        distances = np.append(distances, distance_threshold)

        rna_distribution_dict = calculate_percent_distributions(image_name, image_name_column, structure_1, structure_2, distances, conn)

        rna_distribution_dict.update(image_data_dict)

        distribution_dicts.append(rna_distribution_dict)

    rna_distributions_df = pd.concat([pd.DataFrame(dict_obj) for dict_obj in distribution_dicts])

    return rna_distributions_df

def calculate_distributions_by_image_per_cycle(image_data_list, distance_threshold_dict, step_size, image_name_column, stage_column, structure_1, structure_2, conn):
    """ Takes a list of dictionaries containing image metadata including the image name under the key 'name', a distance threshold, a step_size, the structure_2 distance target, and a postgres db connection object

    Calculates the percent of total RNA and the percent of RNA in granules from 0 microns to the distance threshold away from a structure_2 object (or max image distance if the distance_threshold is set to None) at increments dictated by the step size

    Returns a pandas dataframe object containing the distribution data
    """

    # package import
    import psycopg2
    from psycopg2 import sql
    import numpy as np
    import pandas as pd


    cur = conn.cursor()

    distribution_dicts = []

    for image_data_dict in image_data_list:

        image_name = image_data_dict[image_name_column]
        cycle = image_data_dict[stage_column]
        distance_threshold = distance_threshold_dict[cycle]

        print('Calculating cumulative distributions for ' + image_name)

        distances = np.arange(0, distance_threshold, step_size)
        distances = np.append(distances, distance_threshold)

        rna_distribution_dict = calculate_percent_distributions(image_name, image_name_column, structure_1, structure_2, distances, conn)

        rna_distribution_dict.update(image_data_dict)

        distribution_dicts.append(rna_distribution_dict)

    rna_distributions_df = pd.concat([pd.DataFrame(dict_obj) for dict_obj in distribution_dicts])

    return rna_distributions_df

def save_csv(csv_fn, output_dir, df_to_save):
    """ This function takes two strings as inputs and a pandas dataframe. csv_fn describes the desired filename
    output_dir is the directory to save the csv. df_to_save is a pandas dataframe containing data to save

    This function will not overwrite data

    The function tests if a file exists in the output_dir. If a file exists, it prints a message and does nothing
    If a file does not a exist, the dataframe is saved as a csv

    Returns nothing
    """

    # import packages
    import os
    import pandas as pd


    if os.path.isfile(output_dir + '/' + csv_fn):
        print('Data already saved and will not be saved again')
    else:
        df_to_save.to_csv(output_dir + '/' + csv_fn, index = False)

    return None
