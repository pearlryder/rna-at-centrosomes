# package import for distance measurements


# Python packages
import os
import psycopg2
import multiprocessing as mp
from psycopg2 import sql
import numpy as np

# local packages
from pipeline import measure_distance_by_obj, add_distance_columns



## Define the parameters

structure_measurement_tuples = [('rna', 'centrosomes')]

parallel_processing_bool = True

# database info
database_name = 'resubmission'
db_user = "pearlryder"
db_password = ""
db_host = "localhost"

# Update the scale parameters below for the number of microns per pixel in the xy plane and the z plane
xy_scale = 0.065 # microns per pixel in the xy dimension
z_scale = 0.25   # microns between each z step


for structure_tuple in structure_measurement_tuples:
    structure_1 = structure_tuple[0]
    structure_2 = structure_tuple[1]

    print('Measuring distances between ' + structure_1 + ' and ' + structure_2)

    conn = psycopg2.connect(database=database_name, user=db_user, password=db_password, host=db_host)

    add_distance_columns(structure_1, structure_2, conn)

    conn.close()

    measure_distance_by_obj(structure_1, structure_2, parallel_processing_bool, database_name, db_user, db_password, db_host)
