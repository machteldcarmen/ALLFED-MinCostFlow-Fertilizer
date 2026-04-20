# Data

This folder is where you put the FAOSTAT CSV extracts. They're **not**
committed (see `../.gitignore`). Download them from
<https://www.fao.org/faostat/> -> *Data* -> *Bulk downloads*.

## Files you need

### 1) Production per nutrient

One CSV per nutrient (FAOSTAT -> *Inputs* -> *Fertilizers by Nutrient*):

- `production_Nitrogen_N.csv`
- `production_Phosphorus_P.csv`
- `production_Potassium_K.csv`

Columns the loader expects:
`Area`, `Year`, `Element Code` (optional), `Value`.

### 2) Detailed bilateral trade matrix per nutrient

One CSV per nutrient (FAOSTAT -> *Trade* -> *Fertilizers by Nutrient,
detailed trade matrix*):

- `trade_Nitrogen_N.csv`
- `trade_Phosphorus_P.csv`
- `trade_Potassium_K.csv`

Columns the loader expects:
`Reporter Countries`, `Partner Countries`, `Element Code`
(`5610` = import, `5910` = export), `Y1961` ... `Y2023`, `Value`.

## Shape after loading

```python
from src.preprocessing import load_scenario
scenario = load_scenario(
    production_path="data/production_Nitrogen_N.csv",
    trade_path="data/trade_Nitrogen_N.csv",
    year=2020,
)
# scenario == {"countries": [...], "supply": Series, "demand": Series, "edges": DataFrame}
```
