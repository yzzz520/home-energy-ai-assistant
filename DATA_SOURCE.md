# Data Source

## Ausgrid Solar Home Electricity Data

This project uses the Ausgrid Solar Home half-hourly dataset:

- 300 randomly selected residential solar customers
- Ausgrid electricity network, Sydney region
- 1 July 2010 to 30 June 2013
- Gross-metered rooftop solar generation
- Half-hour consumption and generation records

The dataset was originally published by Ausgrid under the
[Creative Commons Attribution 3.0 Australia](https://creativecommons.org/licenses/by/3.0/au/)
licence in 2014.

Current metadata:

- https://data.gov.au/data/en/dataset/nsw-solar-home-electricty-data

Archive credit used for this project:

- Pierre Haessig, *Ausgrid - Solar home electricity data - archive copy*
- Included notes: `Ausgrid solar home electricity data notes (Aug 2014).pdf`

## Why Raw Data Is Not Committed

The three annual CSV files and related archives are approximately 260 MB.
They are excluded from Git to keep the repository lightweight and to avoid
duplicating a third-party data distribution.

Expected local paths:

```text
Ausgrid_solar_home_data/
├── Solar home 2010-2011.csv
├── Solar home 2011-2012.csv
├── Solar home 2012-2013.csv
├── Ausgrid solar home electricity data notes (Aug 2014).pdf
└── README.md
```

The raw dataset licence is separate from this repository's MIT software
licence. Attribution must be retained when redistributing the data.
