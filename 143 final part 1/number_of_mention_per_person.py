def number_of_mention_per_person(directory):
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
    pandas_bokeh.output_file("C:\\Users\\drill\\Documents\\143 final part 1\\Interactive_Plot_number_or_tweet_per_person.html")
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




