import multiprocessing as mp
import time

if __name__ == '__main__':
    run_time = 15
    sleep_time = .001
    p = mp.Process(target=time.sleep, args=(run_time,))
    p.start()
    i = 0
    while p.is_alive():
        i += 1
    print(f'slept {i}/{run_time / sleep_time}')
    p.close()
