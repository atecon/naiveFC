author = Artur Tarassow
email = atecon@posteo.de
version = 0.9
date = 2020-03-04
description = Simple forecasting methods
tags = C53
min-version = 2019b
data-requirement = needs-time-series-data
gui-main = GUI_naiveFC
label = naive forecast(s)
menu-attachment = MAINWIN/Model/TSModels
public = naiveFC plot_naive_forecasts get_naive_forecasts \
  GUI_naiveFC stack_moving_window_forecasts
menu-only = GUI_naiveFC
help = naiveFC.pdf
sample-script = naiveFC_sample.inp
depends = CvDataSplitter string_utils extra
