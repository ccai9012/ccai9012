# Datasets Reference Guide

This comprehensive guide includes all datasets used in the CCAI9012 starter kits, plus additional similar datasets for extended research and projects.

## Datasets Used in Starter Kits

| Dataset | Category | Module | Description | Direct Link | Size | License |
|---------|----------|---------|-------------|-------------|------|---------|
| **Building Profile & Road Network** | Computer Vision | Module 1 | Image pairs for GAN training - building profiles and corresponding road networks | [GitHub - GANmapper](https://github.com/ualsg/GANmapper) | ~500MB | MIT |
| **Yelp Open Dataset** | Text Analysis | Module 2 | Business reviews, user data, and check-ins for sentiment analysis | [Yelp Dataset](https://www.yelp.com/dataset) | ~10GB | Academic Use |
| **Inside Airbnb Dataset** | Text Analysis | Module 2 | Airbnb listings and reviews data for accommodation analysis | [Inside Airbnb](https://insideairbnb.com/get-the-data/) | Varies by city | CC0 1.0 |
| **Energy Action Plans** | Document Analysis | Module 2 | PDF documents containing tribal energy action plans | [CCHRC](https://cchrc.org/) | ~100MB | Public Domain |
| **Google Street View Imagery** | Computer Vision | Module 3 & 4 | Street-level imagery for urban analysis and perception scoring | [Google Maps API](https://developers.google.com/maps/documentation/streetview) | API-based | Commercial |
| **Webcam Data** | Computer Vision | Module 4 | Real-time webcam feeds for pedestrian behavior analysis | [Skyline Webcams](https://www.skylinewebcams.com/en.html) | Streaming | Varies |
| **California Housing Prices** | Machine Learning | Module 4 | Housing price data for regression analysis | [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) | ~1MB | BSD |
| **German Credit Dataset** | Bias Detection | Module 5 | Credit approval data for fairness analysis | [AIF360](https://aif360.readthedocs.io/en/stable/modules/generated/aif360.datasets.GermanDataset.html) | ~100KB | Public |
| **COMPAS Dataset** | Bias Detection | Module 5 | Criminal risk assessment data for bias auditing | [Kaggle - COMPAS](https://www.kaggle.com/datasets/danofer/compass) | ~50KB | Public |

## Additional Public Datasets

### Urban Planning & Real Estate

| Dataset | Description | Direct Link | Size | License |
|---------|-------------|-------------|------|---------|
| **NYC Property Sales** | Real estate transactions in New York City | [NYC OpenData](https://data.cityofnewyork.us/City-Government/NYC-Citywide-Annualized-Calendar-Sales-Update/w2pb-icbu) | ~500MB | Public |
| **London Housing Data** | UK housing prices and features | [Kaggle - London Housing](https://www.kaggle.com/datasets/justinas/housing-in-london) | ~50MB | CC0 1.0 |
| **Zillow Home Value Data** | US housing market data | [Kaggle - Zillow](https://www.kaggle.com/datasets/paultimothymooney/zillow-house-price-data) | ~2GB | Public |
| **OpenStreetMap Building Data** | Global building footprints and attributes | [OSM Buildings](https://www.openstreetmap.org/) | Varies | ODbL |
| **Microsoft Building Footprints** | Global building footprints from satellite imagery | [GitHub - MS Buildings](https://github.com/Microsoft/GlobalMLBuildingFootprints) | ~100GB | ODbL |

### Review & Text Data

| Dataset | Description | Direct Link | Size | License |
|---------|-------------|-------------|------|---------|
| **Amazon Product Reviews** | Multi-domain product reviews for sentiment analysis | [Kaggle - Amazon Reviews](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews) | ~3GB | Academic |
| **TripAdvisor Hotel Reviews** | Hotel reviews with ratings and locations | [Kaggle - TripAdvisor](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews) | ~100MB | CC0 1.0 |
| **Twitter Sentiment Analysis** | Tweet data with sentiment labels | [Kaggle - Twitter Sentiment](https://www.kaggle.com/datasets/kazanova/sentiment140) | ~200MB | Academic |

### Computer Vision & Street Imagery

| Dataset | Description | Direct Link | Size | License |
|---------|-------------|-------------|------|---------|
| **Mapillary Street View** | Global street-level imagery with semantic segmentation | [Mapillary](https://www.mapillary.com/dataset/vistas) | ~50GB | Commercial |
| **Cityscapes Dataset** | Urban street scenes with semantic annotations | [Cityscapes](https://www.cityscapes-dataset.com/) | ~50GB | Academic |
| **ADE20K Dataset** | Scene parsing dataset with indoor/outdoor scenes | [MIT ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/) | ~3GB | BSD |
| **COCO Dataset** | Object detection and segmentation | [COCO](https://cocodataset.org/) | ~25GB | CC BY 4.0 |

### Pedestrian & Traffic Data

| Dataset | Description | Direct Link | Size | License |
|---------|-------------|-------------|------|---------|
| **MOT Challenge** | Multi-object tracking in pedestrian scenarios | [MOT Challenge](https://motchallenge.net/) | ~10GB | Academic |
| **US Highway Traffic Data** | Federal Highway Administration traffic monitoring and statistics | [FHWA Policy Information](https://www.fhwa.dot.gov/policyinformation/tables/tmasdata/) | ~1GB | Public |
| **NYC Taxi Trip Data** | Taxi trip records for mobility analysis | [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | ~10GB/month | Public |
| **Bike Share Data** | Global bike sharing system data | [Kaggle - Bike Share](https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset) | ~100MB | Public |

### Fairness & Bias Detection

| Dataset | Description | Direct Link | Size | License |
|---------|-------------|-------------|------|---------|
| **Adult Income Dataset** | Census data for income prediction bias analysis | [UCI Adult](https://archive.ics.uci.edu/ml/datasets/adult) | ~5MB | Public |
| **Bank Marketing Dataset** | Marketing campaign data for fairness analysis | [UCI Bank](https://archive.ics.uci.edu/ml/datasets/bank+marketing) | ~5MB | Public |
| **ProPublica COMPAS** | Criminal risk assessment analysis | [ProPublica](https://github.com/propublica/compas-analysis) | ~1MB | Public |
| **Fair Lending Dataset** | Mortgage lending data for discrimination analysis | [FFIEC HMDA](https://ffiec.cfpb.gov/data-publication/) | ~1GB | Public |
| **Chicago Police Data** | Police incident reports for bias analysis | [Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2) | ~2GB | Public |

### Government & Policy Documents

| Dataset | Description | Direct Link | Size | License |
|---------|-------------|-------------|------|---------|
| **EU Law Documents** | European Union legal texts and policies | [EUR-Lex](https://eur-lex.europa.eu/homepage.html?locale=en) | Varies | Public |
| **UN Documents** | United Nations reports and resolutions | [UN Documentation](https://digitallibrary.un.org/) | Varies | Public |

### Environmental & Climate Data

| Dataset | Description | Direct Link | Size      | License |
|---------|-------------|-------------|-----------|---------|
| **NASA Climate Data** | Global climate and weather observations | [NASA Earthdata](https://earthdata.nasa.gov/) | ~100TB    | Public |
| **EPA Air Quality Data** | US air pollution measurements | [EPA AQS](https://www.epa.gov/aqs) | ~10GB     | Public |
| **Copernicus Climate Data Store** | European climate reanalysis data, forecasts, and observations | [CDS Climate Copernicus](https://cds.climate.copernicus.eu/) | Varies    | Free Registration |
| **OpenWeatherMap** | Weather data and forecasts | [OpenWeather API](https://openweathermap.org/api) | API-based | Commercial |
| **Sentinel Satellite Data** | European satellite imagery for environmental monitoring | [Copernicus Hub](https://scihub.copernicus.eu/) | ~100TB    | Free |

## Dataset Usage Guidelines

### Before Using Any Dataset:

1. **Check License Requirements**: Ensure you comply with each dataset's license terms
2. **Verify Data Quality**: Examine data completeness and potential biases
3. **Consider Privacy**: Be aware of personal information and anonymization needs
4. **Cite Properly**: Always provide proper attribution when using datasets
5. **Update Regularly**: Check for newer versions or updates to datasets

### Technical Considerations:

- **Storage**: Large datasets may require cloud storage solutions
- **Processing**: Consider computational requirements for big datasets
- **APIs**: Some datasets require API keys and have rate limits
- **Preprocessing**: Plan for data cleaning and transformation steps
- **Ethics**: Consider the ethical implications of your analysis

### Additional Resources:

- [Google Dataset Search](https://datasetsearch.research.google.com/) - Search engine for datasets
- [Kaggle Datasets](https://www.kaggle.com/datasets) - Community-contributed datasets
- [AWS Open Data](https://registry.opendata.aws/) - Free datasets on AWS
- [Papers with Code Datasets](https://paperswithcode.com/datasets) - Academic datasets with benchmarks
- [Data.gov](https://www.data.gov/) - US government datasets

---

*Last updated: November 2024*
*For questions about dataset usage or suggestions for additions, please submit an issue.*
