For reference these are the total number of observations reported by each sensor type in a 5 minute interval 
from sensor.community.

| Sensor           | Number | %   |
|------------------|--------|-----|
| SDS011           | 24029  | 49% |
| BME280           | 10240  | 21% |
| DHT22            | 9352   | 19% |
| DNMS (Laerm)     | 1082   | 2%  |
| SPS30            | 771    | 2%  |
| PMS5003          | 533    | 1%  |
| BMP280           | 509    | 1%  |
| SHT31            | 500    | 1%  |
| PMS7003          | 371    | 1%  |
| BMP180           | 287    | 1%  |
| SHT30            | 283    | 1%  |
| HTU21D           | 215    | 0%  |
| Radiation Si22G  | 143    | 0%  |
| DS18B20          | 89     | 0%  |
| SHT35            | 20     | 0%  |
| PMS1003          | 20     | 0%  |
| NextPM           | 16     | 0%  |
| GPS-NEO-6M       | 15     | 0%  |
| HPM              | 12     | 0%  |
| PMS3003          | 12     | 0%  |
| SCD30            | 10     | 0%  |
| PPD42NS          | 9      | 0%  |
| Radiation SBM-20 | 8      | 0%  |
| Radiation SBM-19 | 7      | 0%  |
| SDS021           | 6      | 0%  |
| SHT85            | 4      | 0%  |
| SHT11            | 2      | 0%  |
| SHT15            | 1      | 0%  |

## Sensors

Sensor entries look like this:
```yaml
  SDS011:
    name: SDS011
    manufacturer: Nova Fitness
    description: |
      This is the specification for the SDS011. One should not consider all those values for granted as there is quite often a big difference
       between what is said to be possible and what is actually possible (and this is why we are setting up all our experiments).
  
        Output: PM2.5, PM10
        Measuring Range: 0.0-999.9μg/m3
        Response Time 1 second
        Serial Data Output Frequency: 1 time/second
        Particle Diameter Resolution: ≤0.3μm
        Relative Error:10%
        Temperature Range:-20~50°C
      
      Note that the SDS011 humidity working range is 0-70%. Above 70% humidity the readings become unreliable.
      
      From: https://aqicn.org/sensor/sds011/
    references:
      - https://aqicn.org/sensor/sds011/

    observed_properties:
      - P1
      - P2
```

Which should be mostly self explanatory. Note that the keys in observed_properties are those as they appear in the data, `mappedProperties:` contains the mapping to a more generic term which is defined in `generic/properties.yaml`. So for example, `P1` measured by the SDS011 gets mapped by this line:
```
    P2                        : pm2.5
```
to a generic term `pm2.5` that is defined like this in `generic/properties.yml`:
```
  pm2.5:
      label: Particulate matter < 2.5 µm
      sameAs:
        - http://purl.oclc.org/NET/ssnx/cf/cf-property#mass_fraction_of_pm2p5_ambient_aerosol_in_air
        - http://dd.eionet.europa.eu/vocabulary/aq/pollutant/6001
        - https://www.iqair.com/us/newsroom/pm2-5
        - https://www.eea.europa.eu/help/glossary/eea-glossary/pm2.5
```