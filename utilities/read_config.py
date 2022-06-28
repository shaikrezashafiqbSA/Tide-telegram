import configparser
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_config_path(settings='dev'):
    config_path = os.path.join(BASE_DIR, 'config', f'config-{settings}.ini')
    return config_path

def get_config(section,key=None,settings='dev'):
    config_path = get_config_path(settings=settings)
    config = configparser.ConfigParser()
    config.read(config_path)
    if key is not None:
        get = config.get(section,key)
    else:
        try:
            get = {k:float(v) for k,v in config[section].items()}
        except:
            get = {k:str(v) for k,v in config[section].items()}
    return get

def write_config(section, key, value, settings='dev'):
    assert section in ['ACCT', 'tideBot','tides']
    assert type(value) == str
    
    config_path = get_config_path(settings=settings)
    config = configparser.ConfigParser()
    config.read(config_path)
    cfgfile = open(config_path,'w')
    
    config.get(section,key,value)
    config.write(cfgfile)
    cfgfile.close()
    print(f"config file updated for Section: {section}, key: {key}, value: {value}")