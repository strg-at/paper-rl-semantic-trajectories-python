#!/bin/sh

echo "Processing trajectories from 2019-10 to 2019-11"
python bin/glove_preprocessing.py --path data/trajectories_2019-10-01_2019-11-01.parquet --output data/glove_trajectories_2019-10-01_2019-11-01.txt
echo "Processing trajectories from 2019-11 to 2019-12"
python bin/glove_preprocessing.py --path data/trajectories_2019-11-01_2019-12-01.parquet --output data/glove_trajectories_2019-11-01_2019-12-01.txt
echo "Processing trajectories from 2019-12 to 2020-01"
python bin/glove_preprocessing.py --path data/trajectories_2019-12-01_2020-01-01.parquet --output data/glove_trajectories_2019-12-01_2020-01-01.txt
echo "Processing trajectories from 2020-01 to 2020-02"
python bin/glove_preprocessing.py --path data/trajectories_2020-01-01_2020-02-01.parquet --output data/glove_trajectories_2020-01-01_2020-02-01.txt
echo "Processing trajectories from 2020-02 to 2020-03"
python bin/glove_preprocessing.py --path data/trajectories_2020-02-01_2020-03-01.parquet --output data/glove_trajectories_2020-02-01_2020-03-01.txt
echo "Processing trajectories from 2020-03 to 2020-04"
python bin/glove_preprocessing.py --path data/trajectories_2020-03-01_2020-04-01.parquet --output data/glove_trajectories_2020-03-01_2020-04-01.txt
echo "Processing trajectories from 2020-04 to 2020-05"
python bin/glove_preprocessing.py --path data/trajectories_2020-04-01_2020-05-01.parquet --output data/glove_trajectories_2020-04-01_2020-05-01.txt
