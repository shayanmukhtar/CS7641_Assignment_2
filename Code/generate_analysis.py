import travelling_salesman
import six_peeks
import knapsack


def run_all_analysis():
    travelling_salesman.main_10_cities()
    six_peeks.main_20_items()
    knapsack.main_20_items()

def main():
    run_all_analysis()


if __name__ == '__main__':
    main()