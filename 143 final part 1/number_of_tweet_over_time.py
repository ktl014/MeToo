def number_of_tweet_over_time(tweet_month_df):
    import pandas as pd
    assert (isinstance(tweet_month_df, pd.DataFrame))
    assert (('time' in tweet_month_df) and ('tweets') in tweet_month_df)
    from math import pi
    import pandas_bokeh
    import bokeh.plotting as bk
    from bokeh.plotting import figure, show, output_notebook, figure, show
    from bokeh.models import Legend, LinearAxis, HoverTool

    output_notebook()
    pandas_bokeh.output_file("C:\\Users\\drill\\Documents\\143 final part 1\\Interactive_Plot_tweet_trend.html")

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