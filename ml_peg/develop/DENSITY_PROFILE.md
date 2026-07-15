Other thoughts for copper_water_interface density profile tests
1. You should read all the copper_water_interface files.
2. It would be good if z_bins=None and z_bins=np.arange(-2, 35, 0.1) could be reconcilled, i.e. maybe we input a tuple (-2,35,0.1) as default then use this to set z_bins
3. md_water_analysis.py --> don't touch this. Only focus on copper_water_interface tests
