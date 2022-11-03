from pathlib import Path
from util.plot_utils import plot_logs, plot_precision_recall



def main():
    logs_path = Path('output/2_e_2_o')
    logs_paths_list = [logs_path]
    plot_logs(logs_paths_list)

if __name__ == '__main__':
    main()