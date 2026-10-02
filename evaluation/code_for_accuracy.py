
from data_streams.garmin_steps_data import *
from data_streams.phone_steps_data import *
from data_streams.location_data import *
from data_streams.app_usage_data import *
from data_streams.garmin_hr_data import *
from data_streams.wifi_data import *
from data_streams.call_log import *
from collections import defaultdict
from data_streams.lock_unlock_data import *
from data_streams.battery_data import *
from data_streams.activity_data import *
from models.stress_prediction_model import *




def obj_query_1_2(uid, start_time, end_time):
    query_1 = "what is the most common location address of u014 on 2023-08-01?"
    query_2 = "what is the most common location address of test009 on 2024-08-31?"


    start_date = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    end_date = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')

    records = get_location_records(uid, start_time, end_time)


def obj_query_7_8(uid, start_time, end_time):
    query_4 = "what was the discrepancy between garmin steps and phone steps for u010 from 19th may 23 to 26th may 23?"
    query_5 = "what was the discrepancy between garmin steps and phone steps for test006 from 6th oct 24 to 13th oct 24??"


    start_date = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    end_date = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')

    # print(start_date, end_date)

    while start_date < end_date:
        phone_steps = get_phone_steps_stats(uid, start_date.strftime('%Y-%m-%d %H:%M:%S'), (start_date + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'))
        garmin_steps = get_total_garmin_steps(uid, start_date.strftime('%Y-%m-%d %H:%M:%S'), (start_date + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'))
        ph = (phone_steps['total_steps'])
        gh = (garmin_steps['total_steps'])

        print(start_date.strftime('%Y-%m-%d %H:%M:%S') + " : " + str(ph - gh))
        start_date += timedelta(days=1)

    phone_steps = get_phone_steps_stats(uid, start_time, end_time)
    garmin_steps = get_total_garmin_steps(uid, start_time, end_time)
    total_ph  = (phone_steps['total_steps'])
    total_gh = (garmin_steps['total_steps'])
    print("Total discrepency: ", total_ph - total_gh)

def obj_query_11_12(uid, start_time, end_time):
    query_11 = "what was the most used app for test007 every day from 8th august 24 to 12th august 24?"
    query_12 = "what was the most used app for test009 every day from 28th july 24 to 1st august 24??"

    start_date = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    end_date = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')

    # print(start_date, end_date)

    while start_date < end_date:
        app_usage = get_app_usage_blocks(uid, start_date.strftime('%Y-%m-%d %H:%M:%S'), (start_date + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'))
        app_dict = defaultdict(lambda: 0)
        for app in app_usage:
            app_dict[app['app']] += app['duration']
        # print(app_dict)
        print(start_date.strftime('%Y-%m-%d %H:%M:%S') + " : " + max(app_dict, key=app_dict.get))
        start_date += timedelta(days=1)

def obj_query_27_28 (uid, start_time, end_time):
    query_27 = "what is the average daily heart rate of test011 from 11/29/24 to 12/04/24?"
    query_28 = "what is the average daily heart rate of u014 from 06/25/23 to 06/30/23?"

    start_date = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    end_date = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')

    # print(start_date, end_date)
    c = 0
    hr = 0
    while start_date < end_date:
        records = get_garmin_hr(uid, start_date.strftime('%Y-%m-%d %H:%M:%S'), (start_date + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'))
        total_hr = 0

        for rec in records:
            total_hr += rec['heart_rate']
        total_len = len(records)
        if total_len == 0:
            print(start_date.strftime('%Y-%m-%d %H:%M:%S') + " : Missing data")
        else:
            print(start_date.strftime('%Y-%m-%d %H:%M:%S') + " : " + str(total_hr/total_len))
            c += 1
            hr += total_hr/total_len
        start_date += timedelta(days=1)
    print("Average HR: ", hr/c)




def obj_query_33_34 (uid, start_time, end_time):
    query_33 = "how many steps did u012 manage to take on jun 2, 2023?"
    query_34 = "how many steps did u010 manage to take on may 19, 2023?"

    print("Phone Steps", get_phone_steps_stats(uid, start_time, end_time)['total_steps'])
    print("Garmin Steps", get_total_garmin_steps(uid, start_time, end_time)['total_steps'])

def obj_query_45_46 (uid, start_time1, end_time1, start_time2, end_time2):
    query_45 = "which apps did test008 spend the most time using on jul 6 2024? which apps did test008 spend the most time using on jul 3 2024? it might reflect their interests or needs during the work day and relax day."
    query_46 = "which apps did test008 spend the most time using on jun 16 2024?which apps did test008 spend the most time using on jun 13 2024? it might reflect their interests or needs during the work day and relax day."

    app_usage = get_app_usage_blocks(uid, start_time1, end_time1)
    app_dict = defaultdict(lambda: 0)
    if not app_usage:
        print ("No data available")
    else:
        for app in app_usage:
            app_dict[app['app']] += app['duration']
        print(start_time1 + " : " + max(app_dict, key=app_dict.get))

    app_usage = get_app_usage_blocks(uid, start_time2, end_time2)
    app_dict = defaultdict(lambda: 0)
    if not app_usage:
        print("No data available")
    else:
        for app in app_usage:
            app_dict[app['app']] += app['duration']
        print(start_time2 + " : " + max(app_dict, key=app_dict.get))


def obj_query_55_56(uid, start_time, end_time):
    query_55 = "on sep11 2024, how many calls did test006 make while connected to wi-fi? were they multitasking or catching up with someone important?"
    query_56  = "on sep20 2024, how many calls did test007 make while connected to wi-fi? were they multitasking or catching up with someone important?"

    wifi_data = get_wifi_blocks(uid, start_time, end_time)
    total_calls = 0
    for w in wifi_data:
        if(w['wifi_name'] != "not_connected"):
            call_log_stats = get_call_log_stats(uid, w['start_time'], w['end_time'])
            total_calls += call_log_stats['total_calls']
    print("Total calls: ", total_calls)

def obj_query_57_58(uid, start_time, end_time):
    query_61 = "on jun 22 2024, 2024, what was the last app test006 used before locking their phone? maybe it was something soothing before resting."
    query_62 = "on jun 12 2024, 2024, what was the last app test006 used before locking their phone? maybe it was something soothing before resting."

    get_lock_unlock_data = get_lock_unlock_blocks(uid, start_time, end_time)
    l = get_lock_unlock_data[-1]
    if l['state'] == "locked":
        app = get_most_recent_app(uid, l['start_time'])
        print("Last app used before locking: ", app)
    else:
        app = get_most_recent_app(uid, l['start_time'])
        print("Last app used before locking: ", app)



def obj_query_63_64(uid, start_time, end_time):
    query_61 = "on aug 5 2024, what apps did test006 use or whom they call while their phone battery was under 20%? it’s a small yet telling detail about how they prioritized their time and who are important to him"
    query_62 = "on oct 9 2024, what apps did test006 use or whom they call while their phone battery was under 20%? it’s a small yet telling detail about how they prioritized their time and who are important to him"


    battery_records = get_battery_records(uid, start_time, end_time)
    print(battery_records)
    low_battery_blocks=[]
    b = {}
    for rec in battery_records:
        if rec['battery_left'] < 20 and b == {}:
            b['start_time'] = rec['timestamp']
        elif b != {}:
            b['end_time'] = rec['timestamp']
            low_battery_blocks.append(b)
            b = {}
    if b != {}:
        b['end_time'] = end_time
        low_battery_blocks.append(b)
    # print(low_battery_blocks)
    for rec in low_battery_blocks:
        call_log = get_call_log_stats(uid, rec['start_time'], rec['end_time'])
        print("Battery block: ", rec)
        print("Call log stats: ", call_log)
        app = get_app_usage_blocks(uid, rec['start_time'], rec['end_time'])
        distinct_app = []
        for a in app:
            distinct_app.append(a['app'])

        print("App usage: ", set(distinct_app))


def obj_query_71_72(uid, start_time, end_time):
    query_71 = "how many steps did the test006 walk without their phone on 2024-06-12?"
    query_72 = "how many steps did test006 walk without their phone on 2024-06-14?"
    ph = get_phone_steps_stats(uid, start_time, end_time)['total_steps']
    gh = get_total_garmin_steps(uid, start_time, end_time)['total_steps']
    print("Phone Steps", ph)
    print("Garmin Steps", gh)

    print("Garmin Steps without phone", gh - ph)

def obj_query_81_82(uid, start_time, end_time):
    query_81 = "how many steps did u010 walk on 2023-06-28 while their phone was charging?"
    query_82 = "how many steps did test011 walk on 2024-11-22 while their phone was charging?"

    blocks = get_discharging_charging_events(uid, start_time, end_time)
    print(blocks)
    for b in blocks:
        if b['battery_state'] == "charging":
            print("Charging block: ", b)
            print("Phone Steps: ", get_phone_steps_stats(uid, b['start_time'], b['end_time'])['total_steps'])
            print("Garmin Steps: ", get_total_garmin_steps(uid, b['start_time'], b['end_time'])['total_steps'])


def obj_query_99_100(uid, start_time, end_time):
    query_99 = "what was the average heart rate of u010 on 2023-06-11 when the phone ring duration exceeded 30 seconds?"
    query_82 = "what was the average heart rate of test011 on 2024-11-16 when the phone ring duration exceeded 30 seconds?"

    call_records = get_call_log_blocks(uid, start_time, end_time)
    for c in call_records:
        if c['phone_ringing_duration'] > 30:
            print("Call record: ", c)
            end_time = datetime.strptime(c['call_time'], '%Y-%m-%d %H:%M:%S') + timedelta(seconds=c['phone_ringing_duration'])
            end_time = end_time.strftime('%Y-%m-%d %H:%M:%S')
            print(end_time)
            print('Garmin HR: ', get_garmin_hr(uid, c['call_time'], end_time))

    # print("Garmin HR: ", get_garmin_hr(uid,  "2023-05-19 20:50:14", "2023-05-19 20:50:44"))



def obj_query_107_108(uid, start_time, end_time):
    query_107 = "how many times did test007 charge their phone on 2024-08-26?"
    query_108 = "how many times did test007 charge their phone on 2024-10-03?"

    blocks = get_discharging_charging_events(uid, start_time, end_time)
    print("Charging blocks: ", len([b for b in blocks if b['battery_state'] == "charging"]))



def obj_query_111_112(uid, start_time, end_time):
    query_111 = "what are the unique wifi names that test006's phone was connected to on 2024-08-06?"
    query_112 = "what are the unique wifi names that u013's phone was connected to on 2023-06-20?"

    blocks = get_wifi_blocks(uid, start_time, end_time)
    names = []
    for b in blocks:
        print(b)
        names += [b['wifi_name']]

    print(set(names))





def obj_query_143_144(uid, start_time, end_time):
    query_143 = "what was the average step count for test008 on 2024-07-07 between 10:00 am and 12:00 pm?"
    query_144 = "what was the average step count for test011 on 2024-12-03 between 10:00 am and 12:00 pm?"

    ph = get_total_garmin_steps(uid, start_time, end_time)['total_steps']
    gh = get_phone_steps_stats(uid, start_time, end_time)['total_steps']

    print("Phone Steps", ph)
    print("Garmin Steps", gh)

def obj_query_151_152(uid, start_time, end_time):
    query_151 = "what were the total call durations for incoming calls on 2024-06-29 for test009?"
    query_152 = "what were the total call durations for incoming calls on 2023-07-17 for u014??"

    call_records = get_call_log_blocks(uid, start_time, end_time)
    duration = 0
    for rec in call_records:
        if rec['call_type'] == "incoming":
            duration += rec['call_duration']

    print("Total incoming call duration: ", duration)


def obj_query_155_156(uid, start_time, end_time):
    query_155 = "between 2024-09-15 and 2024-09-21, how many times did test006’s phone battery go below 20%?"
    query_156 = "between 2023-05-23 and 2023-05-29, how many times did u009’s phone battery go below 20%?"

    battery_records = get_battery_records(uid, start_time, end_time)
    # print(battery_records)
    low_battery_blocks = []
    b = {}
    for rec in battery_records:
        if rec['battery_left'] < 20:
            b['start_time'] = rec['timestamp']
        elif b != {}:
            b['end_time'] = rec['timestamp']
            low_battery_blocks.append(b)
            b = {}
    if b != {}:
        b['end_time'] = end_time
        low_battery_blocks.append(b)
    print("total times", len(low_battery_blocks))

    count = 0
    for rec in battery_records:
        if rec['battery_left'] < 20:
           count += 1
    print("total times counting each drop", count)


def obj_query_175_156(uid, start_time, end_time):
    query_175 = "what is the highest heart rate recorded for u013 on 2023-06-10?"
    query_176 = "what is the highest heart rate recorded for test011 on 2024-11-23?"

    records = get_garmin_hr(uid, start_time, end_time)
    max_hr = 0
    for rec in records:
        max_hr = max(max_hr, rec['heart_rate'])
    print("Max heart rate: ", max_hr)


def obj_query_177_178(uid, start_time, end_time):
    query_175 = "how many times did test006 exercise from june 15 to june 20 2024?"
    query_176 = "how many times did test009 exercise from august 11 to august 16 2024?"

    records = get_activity_blocks(uid, start_time, end_time)
    # records = get_activity_records(uid, start_time, end_time)
    walk = 0
    cycling = 0
    runnning = 0

    # for r in records:
    #     if "walking" in r['activity']:
    #         walk += 1
    #     if "cycling" in r['activity']:
    #         cycling += 1
    #     if "running" in r['activity']:
    #         runnning += 1

    for r in records:
        if r['activity'] == "walking":
            walk += 1
        elif r['activity'] == "cycling":
            cycling += 1
        elif r['activity'] == "running":
            runnning += 1

    print("Walking: ", walk)
    print("Cycling: ", cycling)
    print("Running: ", runnning)
    print("Total: ", walk + cycling + runnning)



def obj_query_191_192(uid, start_time, end_time):
    query_191 = "did test009 forget to charge their phone any time in the second week of august 2024?"
    query_192 = "did test009 forget to charge their phone any time in the second week of august 2024?"

    battery_records = get_battery_records(uid, start_time, end_time)
    min_battery = 100
    min_battery_time = ""
    for rec in battery_records:
        # print(rec)
        min_battery = min(min_battery, rec['battery_left'])
        if min_battery == rec['battery_left']:
            min_battery_time = rec['timestamp']

    print("Min battery: ", min_battery)
    print("Time: ", min_battery_time)


def obj_query_191_192(uid, start_time, end_time):
    query_191 = "did u010 forget to charge their phone any time in the second week of june 2023?"
    query_192 = "did u013 forget to charge their phone any time in the second week of june 2023?"

    battery_records = get_battery_records(uid, start_time, end_time)
    min_battery = 100
    min_battery_time = ""
    for rec in battery_records:
        # print(rec)
        min_battery = min(min_battery, rec['battery_left'])
        if min_battery == rec['battery_left']:
            min_battery_time = rec['timestamp']

    print("Min battery: ", min_battery)
    print("Time: ", min_battery_time)

def obj_query_199_200(uid, start_time, end_time):
    query_199 = "how much time u012 spent walking during the third week of june 2023?"
    query_200 = "how much time u014 spent walking during the third week of october 2023?"

    blocks = generate_total_activity(uid, start_time, end_time)
    print(blocks)


def obj_query_157_158(uid, start_time, end_time):
    query_157 = "what was the total distance covered by test006 during walking activities on 2024-08-25?"
    query_158 = "what was the total distance covered by test006 during walking activities on 2024-08-25??"

    records = get_activity_blocks(uid, start_time, end_time)
    total_distance = 0
    for rec in records:
        if rec['activity'] == "walking":
            total_distance += get_phone_steps_stats(uid, rec['start_time'], rec['end_time'])['total_distance']

    print("Total distance: ", total_distance)

def obj_query_199_120(uid, start_time, end_time):
    query_199 = "what was the hr value of u010 when they reported their highest stress level on 2023-05-19?"
    query_200 = "what was the hr value of test006 when they reported their highest stress level on 2024-10-06?"

    # get the stress level data
    preds = get_stress_predictions(uid, start_time, end_time)
    max_pred = -1
    max_stress_time = ""
    for p in preds:
        max_pred = max(max_pred, p['stress_probability'])
        if max_pred == p['stress_probability']:
            max_stress_time = p['timestamp']
    print("Max stress level: ", max_pred)
    print("Time: ", max_stress_time)


def obj_query_207_208(uid, start_time, end_time):
    query_207 = "was there any time on the day 10/29/24 that test008 lock.unlock their phone repeatedly during a short period of time?"
    query_208 = "was there any time on the day 12/01/24 that test011 lock.unlock their phone repeatedly during a short period of time?"

    lock_unlock_data = get_lock_unlock_blocks(uid, start_time, end_time)
    for i in range(1, len(lock_unlock_data)):
        start_time = datetime.strptime(lock_unlock_data[i]['start_time'], '%Y-%m-%d %H:%M:%S')
        end_time = datetime.strptime(lock_unlock_data[i]['end_time'], '%Y-%m-%d %H:%M:%S')
        if end_time - start_time < timedelta(minutes=5):
            print("Repeated lock unlock: ", lock_unlock_data[i])

def obj_query_193_194(uid, start_time1, end_time1, start_time2, end_time2):
    query_193 = "was u013 more or less stressed on 6/10 or 6/9 2023 11pm - 12am?"
    query_194 = "was u008 more or less stressed on 4/4 or 4/3 2023 11pm - 12am?"

    preds1 = get_stress_predictions(uid, start_time1, end_time1)
    preds2 = get_stress_predictions(uid, start_time1, end_time1)

    if not preds1:
        print("No data available for ", start_time1, end_time1)
        return
    if not preds2:
        print("No data available for ", start_time2, end_time2)
        return


    mean_stress1 = sum([p['stress_probability'] for p in preds1])/len(preds1)
    mean_stress2 = sum([p['stress_probability'] for p in preds2])/len(preds2)

    if mean_stress1 > mean_stress2:
        print("more stressed on", start_time1, end_time1)
    else:
        print("more stressed on", start_time2, end_time2)

def obj_query_183_184(uid, start_time, end_time):
    query_183 = "what day of the week on the first week of june that u010 has the highest mobility?"
    query_184 = "what day of the week on the first week of october that u014 has the highest mobility?"

    start_date = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    end_date = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')

    # print(start_date, end_date)

    while start_date < end_date:
        phone_steps = get_phone_steps_stats(uid, start_date.strftime('%Y-%m-%d %H:%M:%S'),
                                            (start_date + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'))
        garmin_steps = get_total_garmin_steps(uid, start_date.strftime('%Y-%m-%d %H:%M:%S'),
                                              (start_date + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'))
        activities = generate_total_activity(uid, start_date.strftime('%Y-%m-%d %H:%M:%S'), (start_date + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'))
        ph = (phone_steps['total_steps'])
        gh = (garmin_steps['total_steps'])
        sum = 0
        if 'walking' in activities:
            sum += activities['walking']
        if 'running' in activities:
            sum += activities['running']
        if 'cycling' in activities:
            sum += activities['cycling']
        if 'automotive' in activities:
            sum += activities['automotive']


        print(start_date.strftime('%Y-%m-%d %H:%M:%S'))
        print("Phone Steps: ", ph)
        print("Garmin Steps: ", gh)
        print("Activities: ", sum)
        print(activities)
        start_date += timedelta(days=1)

    # phone_steps = get_phone_steps_stats(uid, start_time, end_time)
    # garmin_steps = get_total_garmin_steps(uid, start_time, end_time)
    # total_ph = (phone_steps['total_steps'])
    # total_gh = (garmin_steps['total_steps'])
    # print("Total discrepency: ", total_ph - total_gh)

def obj_query_77_78(uid, start_time, end_time):
    query_193 = "between 2024-07-03 and 2024-07-06 on which day was test006's stress the highest in the morning (8am-10am)?"
    query_194 = "between 2024-06-29 and 2024-07-02 on which day was test006's stress the highest in the morning (8am-10am)??"

    start_date = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    end_date = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')


    # print(start_date, end_date)
    d = {}
    while start_date < end_date:
        start_time = start_date.strftime('%Y-%m-%d 08:00:00')
        end_time = start_date.strftime('%Y-%m-%d 10:00:00')
        print(start_time, end_time)
        stress = get_stress_predictions(uid, start_time, end_time)
        if stress:
            max_stress = max([s['stress_probability'] for s in stress])
        else:
            max_stress = 0
        d[start_date.strftime('%Y-%m-%d %H:%M:%S')] = max_stress
        print("Max stress: ", max_stress)
        start_date += timedelta(days=1)
    print(d)

def obj_query_23_24(uid, start_time, end_time):
    query_23 = "what was the time when test009 was most active on 21st july 24?"
    query_24 = "what was the time when test009 was most active on 6th june 24?"

    activity = get_activity_blocks(uid, start_time, end_time)
    for a in activity:
        if a['activity'] in ["walking", "running", "cycling"]:
            print(a)


def obj_query_119_120(uid, start_time, end_time):
    query_193 = "what was the hr value of u013 when they reported their highest stress level on 2023-06-07 from 5pm -7pm?"
    query_194 = "what was the hr value of u010 when they reported their highest stress level on 2023-08-08 from 5pm -7pm?"

    preds1 = get_stress_predictions(uid, start_time, end_time)
    max_pred = -1
    max_stress_time = ""
    if not preds1:
        print("No data available for ", start_time, end_time)
        return
    for p in preds1:
        max_pred = max(max_pred, p['stress_probability'])
        if max_pred == p['stress_probability']:
            max_stress_time = p['timestamp']
    print("Max stress level: ", max_pred)
    print("Time: ", max_stress_time)

    max_stress_time_start = datetime.strptime(max_stress_time, '%Y-%m-%d %H:%M:%S') - timedelta(minutes=10)
    max_stress_time_end = datetime.strptime(max_stress_time, '%Y-%m-%d %H:%M:%S') + timedelta(minutes=10)

    print(get_garmin_hr(uid, max_stress_time_start, max_stress_time_end))


def obj_query_135_136(uid, start_time, end_time):
    query_135 = "what is the average resting heart rate of test011 on 2024-11-19??"
    query_136 = "what is the average resting heart rate of u010 on 2023-06-07?"

    act = get_activity_blocks(uid, start_time, end_time)
    c = 0
    hrs = []
    for a in act:
        if a['activity'] == "stationary":
            hr_rec = get_garmin_hr(uid, a['start_time'], a['end_time'])
            if (len(hr_rec) == 0):
                continue
            total_hr = sum([hr['heart_rate'] for hr in hr_rec])
            avg_hr = total_hr/len(hr_rec)
            hrs.append(avg_hr)
            c += 1
    print("Average HR: ", sum(hrs)/c)

    # c = 0
    # total_hr = 0
    # hr_rec = get_garmin_hr(uid, start_time, end_time)
    # for r in hr_rec:
    #     # print(r)
    #     act = get_activity_at_given_time(uid, r['timestamp'])
    #     if act == "stationary":
    #         c += 1
    #         total_hr += r['heart_rate']
    #
    # print("Total stationary hr: ", c)
    # print("Average HR: ", total_hr/c)

def obj_query_123_124(uid, start_time, end_time):
    query_123 = "how many phone apps did test008 use while being stationary on 2024-07-10, and what was the total usage duration?"
    query_124 = "how many phone apps did test006 use while being stationary on 2024-08-07, and what was the total usage duration?"

    act = get_activity_blocks(uid, start_time, end_time)
    durations = {}
    for a in act:
        if a['activity'] == "stationary":
            apps = get_app_usage_blocks(uid, a['start_time'], a['end_time'])
            for app in apps:
                if app['app'] not in durations:
                    durations[app['app']] = 0
                durations[app['app']] += app['duration']

    print("Total apps: ", len(durations))
    print("Total duration: ", sum(durations.values()))
    print (durations)





def main():
    # obj_query_7_8("u013", "2023-06-03 00:00:00", "2023-06-10 23:59:59")
    # obj_query_7_8("u014", "2023-06-04 00:00:00", "2023-06-11 23:59:59")

    # obj_query_11_12("test008", "2024-10-16 00:00:00", "2024-10-20 23:59:59")
    # obj_query_11_12("test009", "2024-07-28 00:00:00", "2024-08-01 23:59:59")

    # obj_query_27_28("test011", "2024-11-29 00:00:00", "2024-12-04 23:59:59")
    # obj_query_27_28("u014", "2023-06-25 00:00:00", "2023-06-30 23:59:59")

    # obj_query_33_34("u012", "2023-06-02 00:00:00", "2023-06-02 23:59:59")
    # obj_query_33_34("u010", "2023-05-19 00:00:00", "2023-05-19 23:59:59")

    # obj_query_45_46("test008", "2024-07-06 00:00:00", "2024-07-06 23:59:59", "2024-07-03 00:00:00", "2024-07-03 23:59:59")
    # obj_query_45_46("test008", "2024-06-16 00:00:00", "2024-06-16 23:59:59", "2024-06-13 00:00:00", "2024-06-13 23:59:59")

    # obj_query_55_56("test006", "2024-09-11 00:00:00", "2024-09-11 23:59:59")
    # obj_query_55_56("test007", "2024-09-20 00:00:00", "2024-09-20 23:59:59")

    # obj_query_57_58("test006", "2024-06-22 00:00:00", "2024-06-22 23:59:59")
    # obj_query_57_58("test006", "2024-07-12 00:00:00", "2024-07-12 23:59:59")

    # obj_query_63_64("test006", "2024-08-05 00:00:00", "2024-08-05 23:59:59")
    # obj_query_63_64("test006", "2024-10-09 00:00:00", "2024-10-09 23:59:59")

    # obj_query_71_72("test006", "2024-06-12 00:00:00", "2024-06-12 23:59:59")
    # obj_query_71_72("test006", "2024-06-14 00:00:00", "2024-06-14 23:59:59")

    # obj_query_81_82("u010", "2023-06-28 00:00:00", "2023-06-28 23:59:59")
    # obj_query_81_82("test011", "2024-11-22 00:00:00", "2024-11-22 23:59:59")

    # obj_query_99_100("u010", "2023-06-11 00:00:00", "2023-06-11 23:59:59")
    # obj_query_99_100("test006", "2024-11-16 00:00:00", "2024-10-16 23:59:59")

    # obj_query_107_108("test007", "2024-08-26 00:00:00", "2024-08-26 23:59:59")
    # obj_query_107_108("test007", "2024-10-03 00:00:00", "2024-10-03 23:59:59")

    # obj_query_111_112("test006", "2024-08-06 00:00:00", "2024-08-06 23:59:59")

    # obj_query_111_112("u013", "2023-06-20 00:00:00", "2023-06-20 23:59:59")

    # obj_query_143_144("test008", "2024-07-07 10:00:00", "2024-07-07 12:00:00")
    # obj_query_143_144("test011", "2024-06-29 10:00:00", "2024-06-29 12:00:00")

    # obj_query_151_152("test009", "2024-06-29 00:00:00", "2024-06-29 23:59:59")
    # obj_query_151_152("u014", "2023-07-17 00:00:00", "2023-07-17 23:59:59")

    # obj_query_155_156("test006", "2024-09-15 00:00:00", "2024-09-21 23:59:59")
    # obj_query_155_156("u009", "2023-05-23 00:00:00", "2023-05-29 23:59:59")

    # obj_query_175_156("u013", "2023-06-10 00:00:00", "2023-06-10 23:59:59")
    # obj_query_175_156("test011", "2024-11-23 00:00:00", "2024-11-23 23:59:59")

    # obj_query_177_178("test006", "2024-06-15 00:00:00", "2024-06-20 23:59:59")
    # obj_query_177_178("test009", "2024-08-11 00:00:00", "2024-08-16 23:59:59")

    # obj_query_183_184("u010", "2023-06-01 00:00:00", "2023-06-07 23:59:59")

    # obj_query_191_192("test009", "2024-06-14 00:00:00", "2023-06-14 23:59:59")
    # obj_query_191_192("test009", "2024-06-07 00:00:00", "2023-06-14 23:59:59")

    # obj_query_199_200("u012", "2023-06-12 00:00:00", "2023-06-18 23:59:59")
    # obj_query_199_200("u014", "2023-10-16 00:00:00", "2023-10-22 23:59:59")

    # obj_query_157_158("test006", "2024-08-25 00:00:00", "2024-08-25 23:59:59")
    # obj_query_157_158("test006", "2024-06-13 00:00:00", "2024-06-13 23:59:59")

    # obj_query_199_120("u010", "2023-05-19 00:00:00", "2023-05-19 23:59:59")
    # obj_query_199_120("test006", "2024-10-06 00:00:00", "2024-10-06 23:59:59")

    # obj_query_207_208("test008", "2024-10-29 00:00:00", "2024-10-29 23:59:59")
    # obj_query_207_208("test011", "2024-12-01 00:00:00", "2024-12-01 23:59:59")

    # obj_query_193_194("u013", "2023-06-10 23:00:00", "2023-06-10 23:59:59", "2023-06-09 23:00:00", "2023-06-09 23:59:59")
    # obj_query_193_194("u008", "2023-04-04 23:00:00", "2023-04-04 23:59:59", "2023-04-03 23:00:00", "2023-04-03 23:59:59")

    # obj_query_183_184("u010", "2023-06-01 00:00:00", "2023-06-07 23:59:59")
    # obj_query_183_184("u014", "2023-10-01 00:00:00", "2023-10-07 23:59:59")

    # obj_query_77_78("test006", "2024-07-03 00:00:00", "2024-07-06 23:59:59")
    # obj_query_77_78("test006", "2024-06-29 00:00:00", "2024-07-02 23:59:59")

    # obj_query_23_24("test009", "2024-07-21 00:00:00", "2024-07-21 23:59:59")
    # obj_query_23_24("test009", "2024-06-06 00:00:00", "2024-06-06 23:59:59")

    # obj_query_119_120("u013", "2023-06-07 17:00:00", "2023-06-07 19:00:00")
    # obj_query_119_120("u010", "2023-08-08 17:00:00", "2023-08-08 19:00:00")

    # obj_query_135_136("test011", "2024-11-19 00:00:00", "2024-11-19 23:59:59")
    # obj_query_135_136("u010", "2023-06-07 00:00:00", "2023-06-07 23:59:59")

    # obj_query_123_124("test008", "2024-07-10 00:00:00", "2024-07-10 23:59:59")
    obj_query_123_124("test006", "2024-08-07 00:00:00", "2024-08-07 23:59:59")



main()
