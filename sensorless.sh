python3 experiment_sequential_ope.py -d 'Sensorless' -m 'NW_regression' -s 2000 -n 20 -r 0.50 -g 0.1 -t 'UCB' -tau 0.7
python3 experiment_sequential_ope.py -d 'Sensorless' -m 'NW_regression' -s 2000 -n 20 -r 0.50 -g 0.1 -t 'RW' -tau 0.7
python3 experiment_sequential_ope.py -d 'Sensorless' -m 'NW_regression' -s 1500 -n 20 -r 1/3 -g 0.1 -t 'UCB' -tau 0.7
python3 experiment_sequential_ope.py -d 'Sensorless' -m 'NW_regression' -s 1500 -n 20 -r 1/3 -g 0.1 -t 'RW' -tau 0.7

python3 experiment_sequential_ope.py -d 'Sensorless' -m 'knn_regression' -s 2000 -n 20 -r 0.50 -g 0.1 -t 'UCB' -tau 0.7
python3 experiment_sequential_ope.py -d 'Sensorless' -m 'knn_regression' -s 2000 -n 20 -r 0.50 -g 0.1 -t 'RW' -tau 0.7
python3 experiment_sequential_ope.py -d 'Sensorless' -m 'knn_regression' -s 1500 -n 20 -r 1/3 -g 0.1 -t 'UCB' -tau 0.7
python3 experiment_sequential_ope.py -d 'Sensorless' -m 'knn_regression' -s 1500 -n 20 -r 1/3 -g 0.1 -t 'RW' -tau 0.7

python3 experiment_sequential_ope.py -d 'Sensorless' -m 'NW_regression' -s 2000 -n 20 -r 0.50 -g 0.1 -t 'TS' -tau 0.7
python3 experiment_sequential_ope.py -d 'Sensorless' -m 'NW_regression' -s 1500 -n 20 -r 1/3 -g 0.1 -t 'TS' -tau 0.7

python3 experiment_sequential_ope.py -d 'Sensorless' -m 'knn_regression' -s 2000 -n 20 -r 0.50 -g 0.1 -t 'TS' -tau 0.7
python3 experiment_sequential_ope.py -d 'Sensorless' -m 'knn_regression' -s 1500 -n 20 -r 1/3 -g 0.1 -t 'TS' -tau 0.7



