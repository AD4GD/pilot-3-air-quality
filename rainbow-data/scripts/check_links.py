import requests
from pathlib import Path
import yaml
import sys


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def check_url(url):
    user_agent = 'Mozilla/20.0.1 (compatible; MSIE 5.5; Windows NT)'
    headers = {'User-Agent': user_agent}
    try:
        r = requests.head(url, headers=headers, stream=True, timeout=1)
        return r.ok, r.status_code
    except Exception as e:
        pass
    try:
        # Use stream = True so that the request does not actually download any data
        r = requests.get(url, headers=headers, stream=True, timeout=2)
        return r.ok, r.status_code
    except Exception as e:
        return False, e


def check_urls_on_keys(file, yaml_list, keys):
    for name, entry in yaml_list.items():
        for key in keys:
            for url in entry.get(key, []):
                print(f"Checking {file} {name}.{key} {url}", end="")
                resolvable, info = check_url(url)
                print(f" {info}")
                if not resolvable:
                    yield (file, name, key, url, info)


properties = load_yaml("rainbow-data/generic/properties.yml")["properties"]
sensors = load_yaml("rainbow-data/sensors.yml")["sensors"]

issues = [
    *check_urls_on_keys("rainbow-data/generic/properties.yml", properties, ["sameAs", "seeAlso"]),
    *check_urls_on_keys("rainbow-data/generic/sensors.yml", properties, ["references"]),
]

if issues:
    for file, name, key, url, info in issues:
        print(f"{file} {name}.{key} {url} {info}", file=sys.stderr)
    sys.exit(112)
