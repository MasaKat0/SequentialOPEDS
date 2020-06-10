#python3 experiment_sequential_ope.py -d 'mnist' -m 'NW_regression' -s 2000 -n 20 -r 0.50 -g 0.1 -t 'UCB'
#python3 experiment_sequential_ope.py -d 'mnist' -m 'NW_regression' -s 2000 -n 20 -r 0.50 -g 0.1 -t 'RW'
python3 experiment_sequential_ope.py -d 'mnist' -m 'NW_regression' -s 1500 -n 20 -r 0.3334 -g 0.1 -t 'UCB'
python3 experiment_sequential_ope.py -d 'mnist' -m 'NW_regression' -s 1500 -n 20 -r 0.3334 -g 0.1 -t 'RW'

#python3 experiment_sequential_ope.py -d 'mnist' -m 'knn_regression' -s 2000 -n 20 -r 0.50 -g 0.1 -t 'UCB'
#python3 experiment_sequential_ope.py -d 'mnist' -m 'knn_regression' -s 2000 -n 20 -r 0.50 -g 0.1 -t 'RW'
python3 experiment_sequential_ope.py -d 'mnist' -m 'knn_regression' -s 1500 -n 20 -r 0.3334 -g 0.1 -t 'UCB'
python3 experiment_sequential_ope.py -d 'mnist' -m 'knn_regression' -s 1500 -n 20 -r 0.3334 -g 0.1 -t 'RW'

python3 experiment_sequential_ope.py -d 'mnist' -m 'NW_regression' -s 2000 -n 20 -r 0.50 -g 0.1 -t 'TS'
python3 experiment_sequential_ope.py -d 'mnist' -m 'NW_regression' -s 1500 -n 20 -r 0.3334 -g 0.1 -t 'TS'

python3 experiment_sequential_ope.py -d 'mnist' -m 'knn_regression' -s 2000 -n 20 -r 0.50 -g 0.1 -t 'TS'
python3 experiment_sequential_ope.py -d 'mnist' -m 'knn_regression' -s 1500 -n 20 -r 0.3334 -g 0.1 -t 'TS'



