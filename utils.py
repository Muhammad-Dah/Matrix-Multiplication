from timeit import timeit
import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt

def time_function(func, *args, reps):
    """
    Passes *args into a function, func, and times it reps 
    times, returns the average time in milliseconds (ms).
    """
    avg_time = timeit(lambda: func(*args), number=reps) / reps

    return avg_time * (10 ** 6)

def time_measurement(func, disc, reps=7, loops_per_run=10000, units='µs'):
    results = np.array([time_function(func, reps=loops_per_run) for _ in range(reps)])
    avg_run_time = np.mean(results)
    print(f'\n{disc}:')

    if 10 ** 3 < avg_run_time < 10 ** 6:
        units = 'ms'
        avg_run_time /= (10 ** 3)

    elif avg_run_time > 10 ** 6:
        units = 's'
        avg_run_time /= (10 ** 6)

    print(f'{avg_run_time:.5f} {units} average time per run (mean of {reps} runs, {loops_per_run} loops each)')
    return avg_run_time


def time_result_to_dict(t):
    avg_time_µs = t.average * 10 ** 6
    best_time_µs = t.best * 10 ** 6
    std_time_µs = t.stdev * 10 ** 6

    def _convert(_x):
        _unit = 'µs'
        if _x > 10 ** 6:
            _unit = 's'
            _x /= (10 ** 6)

        elif _x > 10 ** 3:
            _unit = 'ms'
            _x /= (10 ** 3)

        elif _x < 1:
            _unit = 'ns'
            _x *= (10 ** 3)

        return _unit, _x

    avg_unit, avg_time = _convert(avg_time_µs)
    best_unit, best_time = _convert(best_time_µs)
    std_unit, std_time = _convert(std_time_µs)

    return {'AVG': (avg_unit, avg_time),
            'BEST': (best_unit, best_time),
            'STD': (std_unit, std_time)
            }


def print_time(t):
    dict_result = time_result_to_dict(t)

    avg_unit, avg_time = dict_result['AVG']
    best_unit, best_time = dict_result['BEST']
    std_unit, std_time = dict_result['STD']
    print('')
    print('{:>20s} ({}) {:.1f}'.format('Average time', avg_unit, avg_time))
    print('{:>20s} ({}) {:.1f}'.format('Best time', best_unit, best_time))
    print('{:>20s} ({}) {:.1f}'.format('Std', std_unit, std_time))


def print_time_from_file(res, index):
    fetch_result = lambda res: res[index]['AVG'] + res[index]['BEST'] + res[index]['STD']

    avg_unit, avg_time, best_unit, best_time, std_unit, std_time = fetch_result(res)

    print('')
    print('{:>20s} ({}) {:.1f}'.format('Average time', avg_unit, avg_time))
    print('{:>20s} ({}) {:.1f}'.format('Best time', best_unit, best_time))
    print('{:>20s} ({}) {:.1f}'.format('Std', std_unit, std_time))


def get_summary_table(N_choices, cpu_times, numpy_times, cuda_naive_times, cuda_fast_times):
    data = [[''] * len(N_choices) for _ in range(len(N_choices))]
    index_range = [f'{N:^5d} x {N:^5d}' for N in N_choices]
    columns = ['CPU', 'numpy', 'GPU, naive', 'GPU, fast']

    def fetch_time(index, *args):
        fetch_avg = lambda res: res[index]['AVG']
        result = []
        for lst in args:
            avg_unit, avg_time = fetch_avg(lst)
            result.append(str('{:.1f} ({})'.format(avg_time, avg_unit)))

        return result

    for i in range(len(N_choices)):
        data[i] = fetch_time(i, cpu_times, numpy_times, cuda_naive_times, cuda_fast_times)

    df = pd.DataFrame(data, index=index_range, columns=columns)

    return df

def get_times_same_scale(N_choices, cpu_times, numpy_times, cuda_naive_times, cuda_fast_times, to_scale = 'µs'):
    data = np.zeros((len(N_choices), len(N_choices)), dtype=np.float32)
    columns = ['CPU', 'numpy', 'GPU, naive', 'GPU, fast']
    scales = [ 'ns' , 'µs' , 'ms' , 's' ]
    
    if to_scale not in scales:
        raise ValueError(f'{to_scale} is not in {str(scales)}')
    
    scale_factor = { s : (-3) * (scales.index(to_scale) - i) for i,s in enumerate(scales) }
    
    def fetch_time(index, *args):
        fetch_avg = lambda res: res[index]['AVG']
        result = []
        for lst in args:
            avg_unit, avg_time = fetch_avg(lst)
            result.append(avg_time * (10 ** scale_factor[avg_unit]))

        return result

    for i in range(len(N_choices)):
        data[i] = fetch_time(i, cpu_times, numpy_times, cuda_naive_times, cuda_fast_times)

    return data

def plt_results(N_choices, cpu_times, numpy_times, cuda_naive_times, cuda_fast_times, to_scale = 'µs'):
    objects = ['CPU' , 'numpy', 'GPU, naive', 'GPU, fast']
    args = (N_choices, cpu_times, numpy_times, cuda_naive_times, cuda_fast_times)
    data = get_times_same_scale(*args, to_scale)
    width = 0.8 / len(data)
    Pos = np.array(range(len(data[0]))) 


    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)

    for i in range(len(data)):
        ax.bar(Pos + i * width, data[:, i], width = width)

    plt.xticks(np.arange(len(N_choices)) + 0.35 , [f'{N:^5d} x {N:^5d}' for N in N_choices])

    ax.set_yscale('log')
    plt.ylabel('calculation time')
    plt.title('Computational Time for Matrix multiplication in µs')
    ax.legend(labels=objects)
    plt.show()

def print5():
    print(5)
