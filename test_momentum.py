from sector_momentum import get_historical_momentum
import logging
logging.basicConfig(level=logging.INFO)
result = get_historical_momentum('2026-03-02')
print('Sectors found:', len(result['sectors']))
print('Slots:', result['slots'])
for name, data in list(result['sectors'].items())[:3]:
    print(name, ':', data['snapshots'])
