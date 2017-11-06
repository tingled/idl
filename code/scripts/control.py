#!/usr/bin/env python
import argparse
import boto.ec2
import json
import logging

from time import sleep


class Control():
    def __init__(self, conf_file):
        self.conf_file = conf_file
        self.settings = self._load_conf()
        self.conn = self._load_conn()
        self.instance = self.get_instance()
        self.logger = logging.getLogger(__name__)

    def __getattr__(self, name):
        return self.settings.get(name)

    def _load_conf(self):
        return json.load(open(self.conf_file, 'r'))

    def _load_conn(self):
        return boto.ec2.connect_to_region(self.settings['region'])

    def get_instance(self):
        reservations = self.conn.get_all_instances(
            filters={'instance-id': self.settings['instance-id']}
        )
        assert len(reservations) == 1
        instances = reservations[0].instances
        assert len(instances) == 1
        return instances[0]

    def start_instance(self):
        while self.instance.state != 'running':
            if self.instance.state == 'stopped':
                self.logger.info('starting instance')
                self.instance.start()
            else:
                self.logger.info('instance state: {}. sleeping 5s'.format(
                    self.instance.state
                ))
                sleep(5)
                self.instance.update()

    def stop_instance(self):
        while self.instance.state != 'stopped':
            if self.instance.state == 'running':
                self.logger.info('stopping instance')
                self.instance.stop()
            else:
                self.logger.info('instance state: {}. sleeping 5s'.format(
                    self.instance.state
                ))
                sleep(5)
                self.instance.update()

    def instance_info(self):
        print('instance state: {}'.format(self.instance.state))
        print('instance ip: {}'.format(self.instance.private_ip_address))


def init_logging(args):
    logger = logging.getLogger(__name__)
    level = logging.INFO
    logger.setLevel(level)
    log_format = "%(asctime)s -  %(message)s"
    formatter = logging.Formatter(log_format)

    if args.log_file:
        handler = logging.FileHandler(args.log_file)
    else:
        handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)

    logger.addHandler(handler)


def start_main(args):
    return Control(args.settings).start_instance()


def stop_main(args):
    return Control(args.settings).stop_instance()


def info_main(args):
    return Control(args.settings).instance_info()


def parse_args():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        '-s', '--settings', default='settings.json',
        help='Settings file'
    )
    parent_parser.add_argument('-l', '--log-file', help='output log file')

    sp = parent_parser.add_subparsers(title='commands')
    start = sp.add_parser('start', help='start the instance')
    start.set_defaults(func=start_main)

    stop = sp.add_parser('stop', help='stop the instance')
    stop.set_defaults(func=stop_main)

    info = sp.add_parser('info', help='info about the instance')
    info.set_defaults(func=info_main)

    return parent_parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    init_logging(args)
    args.func(args)
