import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

fontlist = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

path= '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf'
font_name= font_manager.FontProperties(fname=path).get_name()
rc('font', family=font_name)
plt.text(0.3, 0.3, '한글', size=100)


### graph at a glance
def rf_at_once_1(rf_df_2012, rf_df_2013, rf_df_2014, rf_df_2015, rf_df_2016, \
    rf_df_2017, rf_df_2018, rf_df_2019, rf_df_2020, rf_df_2021, rf_df_2022):
    plt.plot(rf_df_2012.index, rf_df_2012.rf_10184100, label='rf_2012_대곡')
    plt.plot(rf_df_2013.index, rf_df_2013.rf_10184100, label='rf_2013_대곡')
    plt.plot(rf_df_2014.index, rf_df_2014.rf_10184100, label='rf_2014_대곡')
    plt.plot(rf_df_2015.index, rf_df_2015.rf_10184100, label='rf_2015_대곡')
    plt.plot(rf_df_2016.index, rf_df_2016.rf_10184100, label='rf_2016_대곡')
    plt.plot(rf_df_2017.index, rf_df_2017.rf_10184100, label='rf_2017_대곡')
    plt.plot(rf_df_2018.index, rf_df_2018.rf_10184100, label='rf_2018_대곡')
    plt.plot(rf_df_2019.index, rf_df_2019.rf_10184100, label='rf_2019_대곡')
    plt.plot(rf_df_2020.index, rf_df_2020.rf_10184100, label='rf_2020_대곡')
    plt.plot(rf_df_2021.index, rf_df_2021.rf_10184100, label='rf_2021_대곡')
    plt.plot(rf_df_2022.index, rf_df_2022.rf_10184100, label='rf_2022_대곡')
    plt.legend()
    plt.show()

def rf_at_once_2(rf_df_2012, rf_df_2013, rf_df_2014, rf_df_2015, rf_df_2016, \
    rf_df_2017, rf_df_2018, rf_df_2019, rf_df_2020, rf_df_2021, rf_df_2022):
    plt.plot(rf_df_2012.index, rf_df_2012.rf_10184110, label='rf_2012_진관')
    plt.plot(rf_df_2013.index, rf_df_2013.rf_10184110, label='rf_2013_진관')
    plt.plot(rf_df_2014.index, rf_df_2014.rf_10184110, label='rf_2014_진관')
    plt.plot(rf_df_2015.index, rf_df_2015.rf_10184110, label='rf_2015_진관')
    plt.plot(rf_df_2016.index, rf_df_2016.rf_10184110, label='rf_2016_진관')
    plt.plot(rf_df_2017.index, rf_df_2017.rf_10184110, label='rf_2017_진관')
    plt.plot(rf_df_2018.index, rf_df_2018.rf_10184110, label='rf_2018_진관')
    plt.plot(rf_df_2019.index, rf_df_2019.rf_10184110, label='rf_2019_진관')
    plt.plot(rf_df_2020.index, rf_df_2020.rf_10184110, label='rf_2020_진관')
    plt.plot(rf_df_2021.index, rf_df_2021.rf_10184110, label='rf_2021_진관')
    plt.plot(rf_df_2022.index, rf_df_2022.rf_10184110, label='rf_2022_진관')
    plt.legend()
    plt.show()

def rf_at_once_3(rf_df_2012, rf_df_2013, rf_df_2014, rf_df_2015, rf_df_2016, \
    rf_df_2017, rf_df_2018, rf_df_2019, rf_df_2020, rf_df_2021, rf_df_2022):
    plt.plot(rf_df_2012.index, rf_df_2012.rf_10184140, label='rf_2012_송정')
    plt.plot(rf_df_2013.index, rf_df_2013.rf_10184140, label='rf_2013_송정')
    plt.plot(rf_df_2014.index, rf_df_2014.rf_10184140, label='rf_2014_송정')
    plt.plot(rf_df_2015.index, rf_df_2015.rf_10184140, label='rf_2015_송정')
    plt.plot(rf_df_2016.index, rf_df_2016.rf_10184140, label='rf_2016_송정')
    plt.plot(rf_df_2017.index, rf_df_2017.rf_10184140, label='rf_2017_송정')
    plt.plot(rf_df_2018.index, rf_df_2018.rf_10184140, label='rf_2018_송정')
    plt.plot(rf_df_2019.index, rf_df_2019.rf_10184140, label='rf_2019_송정')
    plt.plot(rf_df_2020.index, rf_df_2020.rf_10184140, label='rf_2020_송정')
    plt.plot(rf_df_2021.index, rf_df_2021.rf_10184140, label='rf_2021_송정')
    plt.plot(rf_df_2022.index, rf_df_2022.rf_10184140, label='rf_2022_송정')
    plt.legend()
    plt.show()







### graph in a form of grid
def visualize_rf_grid_1(rf_df_2012, rf_df_2013, rf_df_2014, rf_df_2015, rf_df_2016, \
    rf_df_2017, rf_df_2018, rf_df_2019, rf_df_2020, rf_df_2021, rf_df_2022):
    plt.subplot(4,3,1)
    plt.plot(pd.to_datetime(rf_df_2012.ymdhm), rf_df_2012.rf_10184100)
    plt.title('rf_2012_대곡교')
    plt.subplot(4,3,2)
    plt.plot(pd.to_datetime(rf_df_2013.ymdhm), rf_df_2013.rf_10184100)
    plt.title('rf_2013_대곡교')
    plt.subplot(4,3,3)
    plt.plot(pd.to_datetime(rf_df_2014.ymdhm), rf_df_2014.rf_10184100)
    plt.title('rf_2014_대곡교')
    plt.subplot(4,3,4)
    plt.plot(pd.to_datetime(rf_df_2015.ymdhm), rf_df_2015.rf_10184100)
    plt.title('rf_2015_대곡교')

    plt.subplot(4,3,5)
    plt.plot(pd.to_datetime(rf_df_2016.ymdhm), rf_df_2016.rf_10184100)
    plt.title('rf_2016_대곡교')
    plt.subplot(4,3,6)
    plt.plot(pd.to_datetime(rf_df_2017.ymdhm), rf_df_2017.rf_10184100)
    plt.title('rf_2017_대곡교')
    plt.subplot(4,3,7)
    plt.plot(pd.to_datetime(rf_df_2018.ymdhm), rf_df_2018.rf_10184100)
    plt.title('rf_2018_대곡교')
    plt.subplot(4,3,8)
    plt.plot(pd.to_datetime(rf_df_2019.ymdhm), rf_df_2019.rf_10184100)
    plt.title('rf_2019_대곡교')

    plt.subplot(4,3,9)
    plt.plot(pd.to_datetime(rf_df_2020.ymdhm), rf_df_2020.rf_10184100)
    plt.title('rf_2020_대곡교')
    plt.subplot(4,3,10)
    plt.plot(pd.to_datetime(rf_df_2021.ymdhm), rf_df_2021.rf_10184100)
    plt.title('rf_2021_대곡교')
    plt.subplot(4,3,11)
    plt.plot(pd.to_datetime(rf_df_2022.ymdhm), rf_df_2022.rf_10184100)
    plt.title('rf_2022_대곡교')

    plt.show()

def visualize_rf_grid_2(rf_df_2012, rf_df_2013, rf_df_2014, rf_df_2015, rf_df_2016, \
    rf_df_2017, rf_df_2018, rf_df_2019, rf_df_2020, rf_df_2021, rf_df_2022):
    plt.subplot(4,3,1)
    plt.plot(pd.to_datetime(rf_df_2012.ymdhm), rf_df_2012.rf_10184110)
    plt.title('rf_2012_진관교')
    plt.subplot(4,3,2)
    plt.plot(pd.to_datetime(rf_df_2013.ymdhm), rf_df_2013.rf_10184110)
    plt.title('rf_2013_진관교')
    plt.subplot(4,3,3)
    plt.plot(pd.to_datetime(rf_df_2014.ymdhm), rf_df_2014.rf_10184110)
    plt.title('rf_2014_진관교')
    plt.subplot(4,3,4)
    plt.plot(pd.to_datetime(rf_df_2015.ymdhm), rf_df_2015.rf_10184110)
    plt.title('rf_2015_진관교')

    plt.subplot(4,3,5)
    plt.plot(pd.to_datetime(rf_df_2016.ymdhm), rf_df_2016.rf_10184110)
    plt.title('rf_2016_진관교')
    plt.subplot(4,3,6)
    plt.plot(pd.to_datetime(rf_df_2017.ymdhm), rf_df_2017.rf_10184110)
    plt.title('rf_2017_진관교')
    plt.subplot(4,3,7)
    plt.plot(pd.to_datetime(rf_df_2018.ymdhm), rf_df_2018.rf_10184110)
    plt.title('rf_2018_진관교')
    plt.subplot(4,3,8)
    plt.plot(pd.to_datetime(rf_df_2019.ymdhm), rf_df_2019.rf_10184110)
    plt.title('rf_2019_진관교')

    plt.subplot(4,3,9)
    plt.plot(pd.to_datetime(rf_df_2020.ymdhm), rf_df_2020.rf_10184110)
    plt.title('rf_2020_진관교')
    plt.subplot(4,3,10)
    plt.plot(pd.to_datetime(rf_df_2021.ymdhm), rf_df_2021.rf_10184110)
    plt.title('rf_2021_진관교')
    plt.subplot(4,3,11)
    plt.plot(pd.to_datetime(rf_df_2022.ymdhm), rf_df_2022.rf_10184110)
    plt.title('rf_2022_진관교')

    plt.show()

def visualize_rf_grid_3(rf_df_2012, rf_df_2013, rf_df_2014, rf_df_2015, rf_df_2016, \
    rf_df_2017, rf_df_2018, rf_df_2019, rf_df_2020, rf_df_2021, rf_df_2022):
    plt.subplot(4,3,1)
    plt.plot(pd.to_datetime(rf_df_2012.ymdhm), rf_df_2012.rf_10184140)
    plt.title('rf_2012_송정동')
    plt.subplot(4,3,2)
    plt.plot(pd.to_datetime(rf_df_2013.ymdhm), rf_df_2013.rf_10184140)
    plt.title('rf_2013_송정동')
    plt.subplot(4,3,3)
    plt.plot(pd.to_datetime(rf_df_2014.ymdhm), rf_df_2014.rf_10184140)
    plt.title('rf_2014_송정동')
    plt.subplot(4,3,4)
    plt.plot(pd.to_datetime(rf_df_2015.ymdhm), rf_df_2015.rf_10184140)
    plt.title('rf_2015_송정동')

    plt.subplot(4,3,5)
    plt.plot(pd.to_datetime(rf_df_2016.ymdhm), rf_df_2016.rf_10184140)
    plt.title('rf_2016_송정동')
    plt.subplot(4,3,6)
    plt.plot(pd.to_datetime(rf_df_2017.ymdhm), rf_df_2017.rf_10184140)
    plt.title('rf_2017_송정동')
    plt.subplot(4,3,7)
    plt.plot(pd.to_datetime(rf_df_2018.ymdhm), rf_df_2018.rf_10184140)
    plt.title('rf_2018_송정동')
    plt.subplot(4,3,8)
    plt.plot(pd.to_datetime(rf_df_2019.ymdhm), rf_df_2019.rf_10184140)
    plt.title('rf_2019_송정동')

    plt.subplot(4,3,9)
    plt.plot(pd.to_datetime(rf_df_2020.ymdhm), rf_df_2020.rf_10184140)
    plt.title('rf_2020_송정동')
    plt.subplot(4,3,10)
    plt.plot(pd.to_datetime(rf_df_2021.ymdhm), rf_df_2021.rf_10184140)
    plt.title('rf_2021_송정동')
    plt.subplot(4,3,11)
    plt.plot(pd.to_datetime(rf_df_2022.ymdhm), rf_df_2022.rf_10184140)
    plt.title('rf_2022_송정동')

    plt.show()
