'''
  Custom dijkstra cost for unusual stepcharts
'''

import _params, _movement, _stepcharts

scinfo = _stepcharts.SCInfo()

def get_costs(nm):
  move_skillset = _movement.nm_to_moveskillset(nm)
  costs = _params.movement_costs[move_skillset]['costs']
  return costs


def emperor_d17(nm):
  costs = get_costs(nm)
  custom = {
    # 'Inverted feet big': 10,
  }
  costs.update(custom)
  return costs


custom_costs = {
  'Emperor - BanYa D17 arcade': emperor_d17,
}

def get_custom_cost(nm):
  if nm in custom_costs:
    print(f'Using custom cost for {nm}')
    return custom_costs[nm](nm)
  else:
    return None