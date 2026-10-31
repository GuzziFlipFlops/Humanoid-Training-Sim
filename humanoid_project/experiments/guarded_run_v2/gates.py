"""Versioned behavioral gates, never training-return based. Missing evidence fails closed."""
import math


def finite(value):
    return isinstance(value, (float,int)) and math.isfinite(value)


def walking_checks(report):
    checks = {}
    for speed in ('0.5','1.0'):
        ref, cur = report['rows']['reference_'+speed], report['rows']['candidate_'+speed]
        prefix = speed+':'
        # Fixed practical tolerances, recorded before training. Not claims of statistical equivalence.
        margins = {'speed_mae':.05, 'lateral_speed':.03, 'slip':.03,
                   'torso_tilt_deg':3., 'stance_asymmetry':.04,
                   'foot_crossing':.005, 'knee_crossing':.005, 'nonfoot':.005}
        for key, allowance in margins.items():
            checks[prefix+key] = (finite(cur.get(key)) and finite(ref.get(key))
                                  and cur[key] <= ref[key] + allowance)
        checks[prefix+'falls'] = cur['fall_fraction'] <= ref['fall_fraction'] and cur['fall_fraction'] <= .05
        checks[prefix+'samples'] = cur['evaluated_seconds'] >= .9*(report['duration']-3.)
        checks[prefix+'steps'] = cur['valid_step_replicas'] >= .75*report['replicas']
        checks[prefix+'speed_absolute'] = finite(cur['speed_mae']) and cur['speed_mae'] <= .15
    return checks


def stage_checks(report, stage, seed_report=None):
    r = report['rows']['candidate_target']
    if stage=='index': return {}
    target = r['target']
    checks = {
        'target_speed': finite(r['speed_mae']) and r['speed_mae'] <= max(.12,.08*target),
        'falls': r['fall_fraction'] <= .05,
        'enough_prefall': r['evaluated_seconds'] >= .9*(report['duration']-3.),
        'completed_steps': r['valid_step_replicas'] >= .75*report['replicas'],
        'slip': finite(r['slip']) and r['slip'] <= (.25 if target <= 2. else .35),
        'nonfoot': finite(r['nonfoot']) and r['nonfoot'] <= .02,
        'foot_crossing': finite(r['foot_crossing']) and r['foot_crossing'] <= .02,
        'knee_crossing': finite(r['knee_crossing']) and r['knee_crossing'] <= .02,
        'balance': finite(r['stance_asymmetry']) and r['stance_asymmetry'] <= .40,
    }
    if stage=='balance' and seed_report is not None:
        before=seed_report['rows']['candidate_1.0']['stance_asymmetry']
        after=report['rows']['candidate_1.0']['stance_asymmetry']
        checks['walking_balance_improved'] = finite(before) and finite(after) and after <= max(.10,.95*before)
    if stage in ('targets','torso','speed'):
        checks['unreachable_targets'] = finite(r['target_oob']) and r['target_oob'] <= .03
        checks['waist_unreachable'] = finite(r['waist_oob']) and r['waist_oob'] <= .10
    if stage in ('torso','speed'):
        checks['torso_lean'] = finite(r['torso_excess_fraction']) and r['torso_excess_fraction'] <= .10
    if stage=='speed' and target>=3.5:
        checks['running_has_flight'] = finite(r.get('filtered_flight_fraction')) and r['filtered_flight_fraction'] >= .01
    return checks


def judge(reports, stage, seed_reports=None):
    details=[]
    for i,report in enumerate(reports):
        walk=walking_checks(report)
        task=stage_checks(report,stage,None if seed_reports is None else seed_reports[i])
        details.append({'seed':report['seed'],'walking':walk,'stage':task})
    return {'walking_pass':all(all(d['walking'].values()) for d in details),
            'stage_pass':all(all(d['stage'].values()) for d in details),'checks':details}
