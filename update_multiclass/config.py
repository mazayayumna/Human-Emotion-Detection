import yaml

def Config()-> dict:
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config