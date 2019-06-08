def csv_to_dataframe_time_amount(directory):
    '''Return dataframe object with month:#corresponding tweets mapping, given CSV datasets directory path'''
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
    
def number_of_tweet_over_time(tweet_month_df):
    '''Return bokeh object of #tweets over time given dataframe object of month:#tweets mapping'''
    import pandas as pd
    assert (isinstance(tweet_month_df, pd.DataFrame))
    assert (('time' in tweet_month_df) and ('tweets') in tweet_month_df)
    from math import pi
    import pandas_bokeh
    import bokeh.plotting as bk
    from bokeh.plotting import figure, show, output_notebook, figure, show
    from bokeh.models import Legend, LinearAxis, HoverTool

    output_notebook()
    #pandas_bokeh.output_file("C:\\Users\\drill\\Documents\\143 final part 1\\Interactive_Plot_tweet_trend.html")

    tweet_month_df = tweet_month_df.set_index('time')
    tweet_month_df['tweets'] = tweet_month_df['tweets'].astype(float)

    p = tweet_month_df.plot_bokeh(figsize=(1280, 720), kind='line', title='NUMBER OF TWEETS OVER TIME',
                                  ylabel='Number of Tweets',
                                  zooming=False, hovertool=False, show_figure=False, xlabel='')

    p.title.text_font_size = '25pt'
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.xaxis.axis_label_text_font_size = "15pt"
    p.yaxis.axis_label_text_font_size = "15pt"
    p.xaxis.major_label_text_font_size = "15pt"
    p.yaxis.major_label_text_font_size = "15pt"
    p.legend.label_text_font_size = "10pt"
    p.xaxis.major_label_orientation = pi/4

    from bokeh.models import HoverTool
    from bokeh.plotting import figure, output_file, show, ColumnDataSource

    source = ColumnDataSource(data=dict(
        x=[1, 4, 5, 13, 16, 19],
        y=[4268, 3876, 2600, 5849, 1378, 1094],
        desc=['The first accusation against Kevin Spacey lands', 'second annual Women\'s March', 'Lawsuit against Harvey Weinstein',
              'Christine Blasey Ford accused Brett Kavanaugh for sexual assault',
              'Lifetime network airs docuseries that explores the abuse allegations against Chicago R&B superstar R. Kelly',
              'multiple women accused Biden for touching them in a way they felt was inappropriate'],
    ))

    h = HoverTool(mode="vline", names=['events'])

    h.tooltips = [("event", "@desc")]
    p.circle('x', 'y', size=10, source=source, name='events')
    p.add_tools(h)

    pandas_bokeh.show(p)

    return p
    
def number_of_mention_per_person(directory):
    '''Return bokeh object of #mentions/persion in tweets given CSV dataset directory path'''
    import pandas as pd
    import matplotlib.pyplot as plt
    import calendar
    import os
    from math import pi

    assert (isinstance(directory, str) and os.path.isfile(directory)), "directory not exist"
    df = pd.read_csv(directory)

    df_name_trend = pd.DataFrame(columns=['time', 'Spacey', 'Weinstein'])
    for yr in ['2017', '2018', '2019']:
        for mo in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
            if (((int(yr) == 2017) and (int(mo) > 8)) or ((int(yr) == 2019) and (int(mo) < 7)) or int(yr) == 2018):
                temp = df[df['date'].str.contains('%s-%s' % (yr, mo))]['text']
                df_name_trend = df_name_trend.append({'time': yr + ' ' + calendar.month_name[int(mo)],
                                                      'Spacey': temp[temp.str.contains('Spacey')].shape[0],
                                                      'Weinstein': temp[temp.str.contains('Weinstein')].shape[0],
                                                      'Kavanaugh': temp[temp.str.contains('Kavanaugh')].shape[0],
                                                      'Freeman': temp[temp.str.contains('Freeman')].shape[0],
                                                      'R. Kelly': temp[temp.str.contains('R. Kelly')].shape[0],
                                                      'Biden': temp[temp.str.contains('Biden')].shape[0],
                                                      'Trump': temp[temp.str.contains('Trump')].shape[0],
                                                      'Women March': temp[temp.str.contains('March')].shape[0],
                                                      }, ignore_index=True)
    df_name_trend = df_name_trend.set_index('time')

    import pandas_bokeh
    pandas_bokeh.output_notebook()
    #pandas_bokeh.output_file("C:\\Users\\drill\\Documents\\143 final part 1\\Interactive_Plot_number_or_tweet_per_person.html")
    df_name_trend = df_name_trend.astype(float)
    p = df_name_trend.plot_bokeh(figsize=(1280, 720), kind='line', title='NUMBER OF MENTIONS OVER TIME',
                                 ylabel='Number of Mentions',
                                 zooming=False, hovertool=False, show_figure=False, xlabel='')
    p.title.text_font_size = '25pt'
    p.xaxis.axis_label_text_font_size = "15pt"
    p.yaxis.axis_label_text_font_size = "15pt"
    p.xaxis.major_label_text_font_size = "15pt"
    p.yaxis.major_label_text_font_size = "15pt"
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.legend.label_text_font_size = "10pt"
    p.xaxis.major_label_orientation = pi/4
    pandas_bokeh.show(p)

    return p
