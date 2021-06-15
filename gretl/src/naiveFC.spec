author = Artur Tarassow
email = atecon@posteo.de
version = 0.91
date = 2021-06-15
description = Simple forecasting methods
tags = C53
min-version = 2021b
data-requirement = needs-time-series-data
gui-main = GUI_naiveFC
label = naive forecast(s)
menu-attachment = MAINWIN/Model/TSModels
public = naiveFC plot_naive_forecasts get_naive_forecasts \
  GUI_naiveFC
menu-only = GUI_naiveFC
help = naiveFC.pdf
sample-script = naiveFC_sample.inp
depends = CvDataSplitter string_utils extra StrucTiSM
