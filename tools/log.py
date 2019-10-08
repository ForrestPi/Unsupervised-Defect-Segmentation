import os
import datetime


def get_sys_date():
    return datetime.datetime.now().strftime('%Y-%m-%d-h%H')


class Log:
    def __init__(self, log_dir, cfg_name):
        if os.path.exists(log_dir) is not True:
            os.mkdir(log_dir)
        date = get_sys_date()
        f_name = '{}_{}.txt'.format(date, cfg_name)
        f_path = os.path.join(log_dir, f_name)
        self.log = open(f_path, 'w')

    def wr_cfg(self, configs):
        for _dict in configs:
            self.log.write(str(_dict) + ':\n')
            cfg_dict = configs[_dict]
            for item in cfg_dict:
               mes = '    {}: {}'.format(str(item), str(cfg_dict[item]))
               self.log.write(mes + '\n')
        self.log.write('loss:' + '\n')

    def wr_mes(self, mes):
        self.log.write(mes + '\n')

    def close(self):
        self.log.close()
