def csv_to_dataframe_time_amount(directory):
    """convert the csv file of tweets to number of tweet per month
        :param
        directory: the location of the csv file"""
    import pandas as pd
    import calendar
    import os

    assert (isinstance(directory, str) and os.path.isfile(directory)), "directory not exist"

    yrmo = {}
    # create month dictionary
    for yr in ['2017', '2018', '2019']:
        for mo in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
            yrmo['%s-%s', yr, calendar.month_name[int(mo)]] = 0

    # directory = 'C:\\Users\\drill\\OneDrive\\Desktop\\143 final\\tweet_2017_to_2019_with_hashtags.csv'

    for filename in [directory]:
        df = pd.read_csv(filename)
        tweet_dates = df['date']

    for date in tweet_dates:
        mo = (date.split(' ')[0]).split('-')[1]
        yr = (date.split(' ')[0]).split('-')[0]
        yrmo['%s-%s', yr, calendar.month_name[int(mo)]] += 1

    # create dataframe
    tweet_month_df = pd.DataFrame(columns=['time', 'tweets'])

    for yr in ['2017', '2018', '2019']:
        for mo in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
            if (((int(yr) == 2017) and (int(mo) > 8)) or ((int(yr) == 2019) and (int(mo) < 7)) or int(yr) == 2018):
                #             print (yr,'-',calendar.month_name[int(mo)],',',yrmo['%s-%s',yr,calendar.month_name[int(mo)]])
                tweet_month_df = tweet_month_df.append({'time': yr + ' ' + calendar.month_name[int(mo)],
                                                        'tweets': yrmo['%s-%s', yr, calendar.month_name[int(mo)]]},
                                                       ignore_index=True)

    return tweet_month_df